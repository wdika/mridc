# coding=utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/core/classes/modelPT.py

import copy
import inspect
import os
import uuid
from abc import abstractmethod
from os import path
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import hydra
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.utilities import model_summary, rank_zero_only

import mridc.core.optim
import mridc.utils
from mridc import package_info
from mridc.core.classes.common import Model
from mridc.core.connectors.save_restore_connector import SaveRestoreConnector
from mridc.utils import logging
from mridc.utils.app_state import AppState
from mridc.utils.exceptions import MRIDCBaseException
from mridc.utils.get_rank import get_rank, is_global_rank_zero

__all__ = ["ModelPT"]


class ModelPT(LightningModule, Model):
    """Interface for Pytorch-lightning based mridc models"""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        """
        Base class from which all mridc models should inherit

        Internal global flags that determine core functionality of ModelPT.
        _MODEL_IS_RESTORED:
            This flag determines the context of the model - whether the model is currently being
            restored or not.
            -   When set, it can be assumed that the model's will disable all automatic methods -
                setup_training_data(), setup_validation/test_data() and their multi equivalents.
            -   If a model is being restored from a archive file (tarfile), it can be assumed that
                under this context, the cwd is *inside* the tarfile itself.
        _MODEL_RESTORE_PATH:
            A string path to a a file from which the model is being restored.
            This file can either be a PyTorch Lightning Checkpoint, or a archive (tarfile) that contains
            artifact objects.
            If it is an archive file, during restoration, the cwd will be temporarily moved to inside the
            archive itself.

        Parameters
        ----------
        cfg: configuration object. The cfg object should have (optionally) the following sub-configs:
            - train_ds - to instantiate training dataset
            - validation_ds - to instantiate validation dataset
            - test_ds - to instantiate testing dataset
            - optim - to instantiate optimizer with learning rate scheduler
        trainer: Pytorch Lightning Trainer instance
        """
        if trainer is not None and not isinstance(trainer, Trainer):
            raise ValueError(
                "trainer constructor argument must be either None or pytorch_lightning.Trainer. "
                f"But got {type(trainer)} instead."
            )
        super().__init__()

        # set global vars in AppState
        app_state = AppState()

        # Convert config to a DictConfig
        cfg = mridc.utils.model_utils.convert_model_config_to_dict_config(cfg)

        # Convert config to support Hydra 1.0+ instantiation
        cfg = mridc.utils.model_utils.maybe_update_config_version(cfg)

        if "model" in cfg:
            raise ValueError(
                "Creating model config node is forbidden due to collision problem when loading from checkpoint."
            )

        if "target" not in cfg:
            # This is for Jarvis service.
            OmegaConf.set_struct(cfg, False)
            cfg.target = "{0}.{1}".format(self.__class__.__module__, self.__class__.__name__)
            OmegaConf.set_struct(cfg, True)

        if "mridc_version" not in cfg:
            with open_dict(cfg):
                cfg.mridc_version = package_info.__version__

        self._cfg = cfg

        # init mapping submodule attribute -> config_field for nested MRIDC models
        self._mridc_submodule_name_to_config_field: Dict = dict()

        self.save_hyperparameters("cfg")
        self._train_dl = None
        self._validation_dl = None
        self._test_dl = None
        self._optimizer_param_groups = None
        self._optimizer = None
        self._scheduler = None
        self.set_trainer(trainer)

        self._save_restore_connector = SaveRestoreConnector()

        self._set_model_guid()

        # Set device_id in AppState
        if torch.cuda.is_available() and torch.cuda.current_device() is not None:
            app_state.device_id = torch.cuda.current_device()

        if self._cfg is not None and not self._is_model_being_restored():
            # Setup data loaders now (default) or defer setup to `self.setup()` if `defer_setup` is set in the config
            # of the corresponding dataloader.
            if (
                "train_ds" in self._cfg
                and self._cfg.train_ds is not None
                and not self._cfg.train_ds.get("defer_setup", False)
            ):
                self.setup_training_data(self._cfg.train_ds)

            if (
                "validation_ds" in self._cfg
                and self._cfg.validation_ds is not None
                and not self._cfg.validation_ds.get("defer_setup", False)
            ):
                self.setup_multiple_validation_data(val_data_config=cfg.validation_ds)

            if (
                "test_ds" in self._cfg
                and self._cfg.test_ds is not None
                and not self._cfg.test_ds.get("defer_setup", False)
            ):
                self.setup_multiple_test_data(test_data_config=cfg.test_ds)

        else:
            if "train_ds" in self._cfg and self._cfg.train_ds is not None:  # type: ignore
                logging.warning(
                    "If you intend to do training or fine-tuning, please call the ModelPT.setup_training_data() "
                    "method and provide a valid configuration file to setup the train data loader.\n"
                    # type: ignore
                    f"Train config : \n{OmegaConf.to_yaml(self._cfg.train_ds)}"
                )
            if "validation_ds" in self._cfg and self._cfg.validation_ds is not None:  # type: ignore
                logging.warning(
                    "If you intend to do validation, please call the ModelPT.setup_validation_data() or "
                    "ModelPT.setup_multiple_validation_data() method and provide a valid configuration file to "
                    "setup the validation data loader(s). \n"
                    # type: ignore
                    f"Validation config : \n{OmegaConf.to_yaml(self._cfg.validation_ds)}"
                )
            if "test_ds" in self._cfg and self._cfg.test_ds is not None:  # type: ignore
                logging.warning(
                    "Please call the ModelPT.setup_test_data() or ModelPT.setup_multiple_test_data() method "
                    "and provide a valid configuration file to setup the test data loader(s).\n"
                    # type: ignore
                    f"Test config : \n{OmegaConf.to_yaml(self._cfg.test_ds)}"
                )

        # ModelPT wrappers over subclass implementations
        self._training_step = mridc.utils.model_utils.wrap_training_step(self.training_step)  # type: ignore

    def __init_subclass__(cls) -> None:
        """This method is called when a subclass is created."""
        cls._save_restore_connector = SaveRestoreConnector()

    def register_artifact(self, config_path: str, src: str, verify_src_exists: bool = True):
        """
        Register model artifacts with this function. These artifacts (files) will be included inside .mridc file when
        model.save_to("model.mridc") is called.

        How it works:

            1. It always returns existing absolute path which can be used during Model constructor call EXCEPTION: \
            src is None or "" in which case nothing will be done and src will be returned
            2. It will add (config_path, model_utils.ArtifactItem()) pair to self.artifacts

            .. code-block::
                    If "src" is local existing path:
                        then it will be returned in absolute path form.
                    elif "src" starts with "mridc_file:unique_artifact_name":
                        .mridc will be untarred to a temporary folder location and an actual existing path will be
                        returned
                    else:
                        an error will be raised.

        If "src" is local existing path, then it will be returned in absolute path form.
        elif "src" starts with "mridc_file:unique_artifact_name" .mridc will be untarred to a temporary folder \
        location and an actual existing path will be returned else an error will be raised.

        WARNING: use .register_artifact calls in your models' constructors.
        The returned path is not guaranteed to exist after you have exited your model's constructor.

        Parameters
        ----------
        config_path: Artifact key. Usually corresponds to the model config.
        src: Path to artifact.
        verify_src_exists: If set to False, then the artifact is optional and register_artifact will return None \
        even if src is not found. Defaults to True.

        Returns
        -------
        If src is not None or empty it always returns absolute path which is guaranteed to exist during model \
        instance life.
        """
        if src is None or not src:
            return src

        if Path(src).suffix == ".mridc":
            raise MRIDCBaseException(
                "Registering .mridc files as artifacts not supported. "
                "If you are trying to make a nested model, use `register_mridc_submodule`."
            )

        if not hasattr(self, "artifacts"):
            self.artifacts: Dict[str, mridc.utils.model_utils.ArtifactItem] = {}

        if self.artifacts is None:
            self.artifacts = {}

        if config_path in self.artifacts:
            logging.warning(
                f"You tried to register an artifact under config key={config_path} but an artifact for "
                "it has already been registered."
            )

        return self._save_restore_connector.register_artifact(self, config_path, src, verify_src_exists)

    def has_artifacts(self) -> bool:
        """Returns True if model has artifacts registered."""
        return hasattr(self, "artifacts") and self.artifacts is not None and len(self.artifacts) > 0

    def has_native_or_submodules_artifacts(self) -> bool:
        """Returns True if it has artifacts or any of the submodules have artifacts."""
        for module in self.modules():
            if (
                isinstance(module, ModelPT)
                and hasattr(module, "artifacts")
                and module.artifacts is not None
                and len(module.artifacts) > 0
            ):
                return True
        return False

    def has_mridc_submodules(self) -> bool:
        """Returns True if it has any registered MRIDC submodules."""
        return len(self._mridc_submodule_name_to_config_field) > 0

    def register_mridc_submodule(self, name: str, config_field: str, model: "ModelPT") -> None:
        """
        Adds a MRIDC model as a submodule.

        Submodule can be accessed via the `name` attribute on the parent MRIDC model this submodule was registered on
        (`self`).

        In the saving process, the whole parent model (self) is held as a solid model with artifacts from the child
        submodule, the submodule config will be saved to the `config_field` of the parent model.

        This method is necessary to create a nested model, e.g.
            .. code-block:: python

                class ParentModel(ModelPT):
                    def __init__(self, cfg, trainer=None):
                        super().__init__(cfg=cfg, trainer=trainer)

                        # annotate type for autocompletion and type checking (optional)
                        self.child_model: Optional[ChildModel] = None
                        if cfg.get("child_model") is not None:
                            self.register_mridc_submodule(
                                name="child_model",
                                config_field="child_model",
                                model=ChildModel(self.cfg.child_model, trainer=trainer),
                            )
                        # ... other code

        Parameters
        ----------
        name : str
            Name of the submodule. This name will be used to access the submodule from the parent model.
        config_field : str
            Name of the config field where the submodule config will be saved.
        model : ModelPT
            The submodule model.
        """
        # check it is a real MRIDC model
        if not isinstance(model, ModelPT):
            raise MRIDCBaseException(
                f"Model is not and instance of ModelPT, so can't be registered. Got {type(model).__name__}"
            )
        # check if it is called after __init__
        if not hasattr(self, "_mridc_submodule_name_to_config_field"):
            raise MRIDCBaseException(
                "You are trying to register a submodule before the model is initialized. This is not allowed. "
                "Did you forget to call `super().__init__`?"
            )
        # assign attribute to self
        setattr(self, name, model)
        # add to the submodules mapping
        self._mridc_submodule_name_to_config_field[name] = config_field

    def named_mridc_modules(
        self, prefix_name: str = "", prefix_config: str = ""
    ) -> Iterator[Tuple[str, str, "ModelPT"]]:
        """
        Returns an iterator over all MRIDC submodules recursively, yielding tuples of (attribute path, path in config,
        submodule), starting from the core module.

        Parameters
        ----------
        prefix_name : str
            Prefix for the name path.
        prefix_config : str
            Prefix for the path in config.

        Returns
        -------
        Iterator over (attribute path, path in config, submodule), starting from (prefix, self).
        """
        if not hasattr(self, "_mridc_submodule_name_to_config_field"):
            raise MRIDCBaseException(
                "Model is not fully initialized. Calling `named_mridc_modules` before __init__ not allowed. "
                "Did you forget to call `super().__init__`?"
            )

        yield prefix_name, prefix_config, self

        # recursive iteration over all MRIDC submodules
        for name, config_field in self._mridc_submodule_name_to_config_field.items():
            attribute_path = f"{prefix_name}.{name}" if prefix_name else name
            config_path = f"{prefix_config}.{config_field}" if prefix_config else config_field
            module: ModelPT = getattr(self, name)
            for submodule_name, subconfig_path, submodule in module.named_mridc_modules(
                prefix_name=attribute_path, prefix_config=config_path
            ):
                yield submodule_name, subconfig_path, submodule

    def save_to(self, save_path: str):
        """
        Saves model instance (weights and configuration) into .mridc file. You can use "restore_from" method to fully
        restore instance from .mridc file. .mridc file is an archive (tar.gz) with the following:
        - model_config.yaml - model configuration in .yaml format. You can deserialize this into cfg argument for \
         model's constructor
        - model_wights.ckpt - model checkpoint

        Parameters
        ----------
        Path to .mridc file where model instance should be saved.
        """

        def maybe_make_save_dir(_path: "Path"):
            """Creates directory if it does not exist"""
            if not _path.parent.exists():
                _path.parent.mkdir(parents=True)

        save_path = Path(save_path).expanduser().resolve()  # type: ignore
        app_state = AppState()
        if app_state.model_parallel_size is not None:
            if app_state.model_parallel_size > 1 and type(self._save_restore_connector) is SaveRestoreConnector:
                raise ValueError(
                    "Default mridc SaveRestoreConnector will not work in model parallel mode. You should use a "
                    "connector which supports model parallel mode. You can also use a custom one."
                )
            if app_state.data_parallel_rank == 0:
                maybe_make_save_dir(Path(save_path))
            # connector checks for ranks properly, no need to check here
            # downstream tasks expect str, not Path
            self._save_restore_connector.save_to(self, str(save_path))
        elif is_global_rank_zero():
            maybe_make_save_dir(Path(save_path))
            # downstream tasks expect str, not Path
            self._save_restore_connector.save_to(self, str(save_path))

    @classmethod
    def restore_from(  # type: ignore
        cls,
        restore_path: str,
        override_config_path: Optional[Union[OmegaConf, str]] = None,
        map_location: Optional[torch.device] = None,
        strict: bool = True,
        return_config: bool = False,
        save_restore_connector: SaveRestoreConnector = None,
        trainer: Optional[Trainer] = None,
    ):
        """
        Restores model instance (weights and configuration) from .mridc file.

        Parameters
        ----------
        restore_path: path to .mridc file from which model should be instantiated override_config_path: path to a \
        yaml config that will override the internal config file or an OmegaConf/DictConfig object representing the \
        model config.
        map_location: Optional torch.device() to map the instantiated model to a device. By default (None), it will \
        select a GPU if available, falling back to CPU otherwise.
        strict: Passed to load_state_dict. By default, True.
        return_config: If set to true, will return just the underlying config of the restored model as an \
        OmegaConf/DictConfig object without instantiating the model.
        trainer: Optional, a pytorch lightning Trainer object that will be forwarded to the instantiated model's \
        constructor.
        save_restore_connector: Can be overridden to add custom save and restore logic.

        Example
        -------

        .. code-block::

            model = mridc.collections.asr.models.EncDecCTCModel.restore_from('asr.mridc')
            assert isinstance(model, mridc.collections.asr.models.EncDecCTCModel)


        Returns
        -------
        An instance of type cls or its underlying config (if return_config is set).
        """
        if save_restore_connector is None:
            save_restore_connector = SaveRestoreConnector()

        if save_restore_connector.model_extracted_dir is None:
            restore_path = os.path.abspath(os.path.expanduser(restore_path))
        else:
            restore_path = os.path.abspath(os.path.expanduser(save_restore_connector.model_extracted_dir))

        if not path.exists(restore_path):
            raise FileNotFoundError(f"Can't find {restore_path}")

        app_state = AppState()
        app_state.model_restore_path = restore_path

        cls.update_save_restore_connector(save_restore_connector)
        instance = cls._save_restore_connector.restore_from(
            cls, restore_path, override_config_path, map_location, strict, return_config, trainer
        )
        if isinstance(instance, ModelPT):
            instance._save_restore_connector = save_restore_connector
        return instance

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        *args,
        map_location: Optional[Union[Dict[str, str], str, torch.device, int, Callable]] = None,
        hparams_file: Optional[str] = None,
        strict: bool = True,
        **kwargs,
    ):
        """
        Loads ModelPT from checkpoint, with some maintenance of restoration.
        For documentation, please refer to LightningModule.load_from_checkpoint() documentation.
        """
        checkpoint = None

        try:
            cls._set_model_restore_state(is_being_restored=True)

            checkpoint = super().load_from_checkpoint(
                checkpoint_path=checkpoint_path,
                *args,  # type: ignore
                map_location=map_location,
                hparams_file=hparams_file,
                strict=strict,
                **kwargs,
            )

        finally:
            cls._set_model_restore_state(is_being_restored=False)
        return checkpoint

    @abstractmethod
    def setup_training_data(self, train_data_config: Union[DictConfig, Dict]):
        """Setups data loader to be used in training."""

    @abstractmethod
    def setup_validation_data(self, val_data_config: Union[DictConfig, Dict]):
        """Setups data loader to be used in validation."""

    def setup_test_data(self, test_data_config: Union[DictConfig, Dict]):
        """(Optionally) Setups data loader to be used in test."""
        raise NotImplementedError()

    def setup_multiple_validation_data(self, val_data_config: Union[DictConfig, Dict]):
        """(Optionally) Setups data loader to be used in validation."""
        # Set some placeholder overridden by helper method
        self._val_dl_idx = 0
        self.validation_names = None

        # preserve config
        self._update_dataset_config(dataset_name="validation", config=val_data_config)

        try:
            self._multi_dataset_mode = True
            mridc.utils.model_utils.resolve_validation_dataloaders(model=self)
        finally:
            self._multi_dataset_mode = False

        if (
            self.validation_names is None
            and self._validation_dl is not None
            and type(self._validation_dl) in [list, tuple]
        ):
            self.validation_names = [f"val_{idx}_" for idx in range(len(self._validation_dl))]

    def setup_multiple_test_data(self, test_data_config: Union[DictConfig, Dict]):
        """(Optionally) Setups data loader to be used in test, with support for multiple data loaders."""
        # Set some placeholder overridden by helper method
        self._test_dl_idx = 0
        self.test_names = None
        self._test_dl = None  # type: ignore

        # preserve config
        self._update_dataset_config(dataset_name="test", config=test_data_config)

        try:
            self._multi_dataset_mode = True
            mridc.utils.model_utils.resolve_test_dataloaders(model=self)
        finally:
            self._multi_dataset_mode = False

        if self.test_names is None and self._test_dl is not None and type(self._test_dl) in [list, tuple]:
            self.test_names = [f"test_{idx}_" for idx in range(len(self._test_dl))]

    def setup_optimization(self, optim_config: Optional[Union[DictConfig, Dict]] = None):
        """
        Prepares an optimizer from a string name and its optional config parameters.

        Parameters
        ----------
        optim_config: A dictionary containing the following keys:
            - lr: mandatory key for learning rate. Will raise ValueError if not provided.
            - optimizer: string name pointing to one of the available optimizers in the registry. If not provided, \
            defaults to "adam".
            - opt_args: Optional list of strings, in the format "arg_name=arg_value". The list of "arg_value" will \
            be parsed and a dictionary of optimizer kwargs will be built and supplied to instantiate the optimizer.

        Returns
        -------
        An instance of an optimizer.
        """
        # Setup the optimizer parameter groups (by default use all parameters that are trainable).
        self.setup_optimizer_param_groups()

        # If config was not explicitly provided, use default
        if optim_config is None and self._cfg is not None and hasattr(self._cfg, "optim"):
            optim_config = self._cfg.optim

        # If config is still None, or internal config has no Optim, return without instantiation
        if optim_config is None:
            logging.info("No optimizer config provided, therefore no optimizer was created")
            return
        # Preserve the configuration
        if not isinstance(optim_config, DictConfig):
            optim_config = OmegaConf.create(optim_config)

        # See if internal config has `optim` namespace before preservation
        if self._cfg is not None and hasattr(self._cfg, "optim"):
            if self._cfg.optim is None:
                self._cfg.optim = copy.deepcopy(optim_config)
            else:
                with open_dict(self._cfg.optim):
                    self._cfg.optim = copy.deepcopy(optim_config)

        # Setup optimizer and scheduler
        if optim_config is not None and isinstance(optim_config, DictConfig):
            optim_config = OmegaConf.to_container(optim_config, resolve=True)

        if self._trainer is None:
            logging.warning("Trainer wasn't specified in model constructor. Make sure that you really wanted it.")

        if "sched" in optim_config and self._trainer is not None:
            if not isinstance(self._trainer.accumulate_grad_batches, int):
                raise ValueError("We do not currently support gradient accumulation that is not an integer.")
            if self._trainer.max_steps is None or self.trainer.max_steps < 0:  # type: ignore
                # Store information needed to calculate max_steps
                optim_config["sched"]["t_max_epochs"] = self._trainer.max_epochs
                optim_config["sched"]["t_accumulate_grad_batches"] = self._trainer.accumulate_grad_batches
                optim_config["sched"]["t_limit_train_batches"] = self._trainer.limit_train_batches
                optim_config["sched"]["t_num_workers"] = self._trainer.num_devices * self._trainer.num_nodes
            else:
                optim_config["sched"]["max_steps"] = self._trainer.max_steps

        # Force into DictConfig from nested structure
        optim_config = OmegaConf.create(optim_config)
        # Get back nested dict so we its mutable
        optim_config = OmegaConf.to_container(optim_config, resolve=True)

        # Extract scheduler config if inside optimizer config
        if "sched" in optim_config:
            scheduler_config = optim_config.pop("sched")
        else:
            scheduler_config = None

        # Check if caller provided optimizer name, default to Adam otherwise
        optimizer_cls = optim_config.get("_target_", None)

        if optimizer_cls is None:
            # Try to get optimizer name for dynamic resolution, defaulting to Adam
            optimizer_name = optim_config.get("name", "adam")
        elif inspect.isclass(optimizer_cls):
            optimizer_name = optimizer_cls.__name__.lower()
        else:
            # resolve the class name (lowercase) from the class path if not provided
            optimizer_name = optimizer_cls.split(".")[-1].lower()

        # We are guaranteed to have lr since it is required by the argparser
        # But maybe user forgot to pass it to this function
        lr = optim_config.get("lr", None)

        # Check if caller has optimizer kwargs, default to empty dictionary
        if "args" in optim_config:
            optimizer_args = optim_config.pop("args")
            optimizer_args = mridc.core.optim.optimizers.parse_optimizer_args(optimizer_name, optimizer_args)
        else:
            optimizer_args = copy.deepcopy(optim_config)

            # Remove extra parameters from optimizer_args nest
            # Assume all other parameters are to be passed into optimizer constructor
            optimizer_args.pop("name", None)
            optimizer_args.pop("cls", None)
            optimizer_args.pop("lr", None)

        # Adaptive schedulers don't need `lr`
        if lr is not None:
            optimizer_args["lr"] = lr

            # Actually instantiate the optimizer
            if optimizer_cls is None:
                optimizer = mridc.core.optim.optimizers.get_optimizer(optimizer_name)
                optimizer = optimizer(self._optimizer_param_groups, **optimizer_args)

                logging.info("Optimizer config = %s", str(optimizer))

                self._optimizer = optimizer  # type: ignore

            elif inspect.isclass(optimizer_cls):
                optimizer = optimizer_cls(self._optimizer_param_groups, **optimizer_args)
                logging.info("Optimizer config = %s", str(optimizer))

                self._optimizer = optimizer  # type: ignore

            else:
                # Attempt class path resolution
                try:
                    optimizer_cls = OmegaConf.create({"_target_": optimizer_cls})
                    optimizer_config = {"lr": lr} if lr is not None else {}
                    optimizer_config |= optimizer_args

                    optimizer_instance = hydra.utils.instantiate(
                        optimizer_cls, self._optimizer_param_groups, **optimizer_config
                    )  # type: DictConfig

                    logging.info("Optimizer config = %s", str(optimizer_instance))

                    self._optimizer = optimizer_instance

                except Exception as e:
                    logging.error(
                        f"Could not instantiate class path - {optimizer_cls} with kwargs {str(optimizer_config)}"
                    )

                    raise e

            # print(f"scheduler_config = {scheduler_config}")
            if isinstance(scheduler_config["name"], list):
                _schedulers = [
                    mridc.core.optim.lr_scheduler.prepare_lr_scheduler(
                        optimizer=self._optimizer,
                        scheduler_config={
                            "name": scheduler_config["name"][i],
                            "min_lr": scheduler_config["min_lr"][i],
                            "last_epoch": scheduler_config["last_epoch"][i],
                            "warmup_ratio": scheduler_config["warmup_ratio"][i],
                            "monitor": scheduler_config["monitor"][i],
                            "t_max_epochs": scheduler_config["t_max_epochs"],
                            "t_accumulate_grad_batches": scheduler_config["t_accumulate_grad_batches"],
                            "t_limit_train_batches": scheduler_config["t_limit_train_batches"],
                            "t_num_workers": scheduler_config["t_num_workers"],
                        },
                        train_dataloader=self._train_dl,
                    )
                    for i in range(len(scheduler_config["name"]))
                ]

                self._scheduler = _schedulers  # type: ignore
                self._optimizer = [self._optimizer] * len(scheduler_config["name"])  # type: ignore
            else:
                # Try to instantiate scheduler for optimizer
                self._scheduler = mridc.core.optim.lr_scheduler.prepare_lr_scheduler(  # type: ignore
                    optimizer=self._optimizer, scheduler_config=scheduler_config, train_dataloader=self._train_dl
                )

            # Return the optimizer with/without scheduler
            # This return allows multiple optimizers or schedulers to be created
            return self._optimizer, self._scheduler

    def setup_optimizer_param_groups(self):
        """
        Used to create param groups for the optimizer. As an example, this can be used to specify per-layer learning
        rates:

        .. code-block::

            optim.SGD([
                        {'params': model.base.parameters()},
                        {'params': model.classifier.parameters(), 'lr': 1e-3}
                        ], lr=1e-2, momentum=0.9)

        See https://pytorch.org/docs/stable/optim.html for more information. By default, ModelPT will use
        self.parameters(). Override this method to add custom param groups.
        """
        param_groups = None
        if hasattr(self, "parameters"):
            param_groups = [{"params": self.parameters()}]
        self._optimizer_param_groups = param_groups

    def configure_optimizers(self):
        """Configure optimizers and schedulers for training."""
        self.setup_optimization()

        if isinstance(self._scheduler, list) and self._scheduler[0] is None:
            return self._optimizer

        if self._scheduler is None:
            return self._optimizer

        if isinstance(self._optimizer, list):
            return self._optimizer, self._scheduler

        return [self._optimizer], [self._scheduler]

    def setup(self, stage: Optional[str] = None):
        """
        Called at the beginning of fit, validate, test, or predict. This is called on every process when using DDP.

        Parameters
        ----------
        stage: str
            fit, validate, test or predict
        """
        if stage == "fit":
            train_deferred_setup = (
                "train_ds" in self._cfg
                and self._cfg.train_ds is not None
                and self._cfg.train_ds.get("defer_setup", False)
            )
            if self.train_dataloader() is None and train_deferred_setup:
                self.setup_training_data(self._cfg.train_ds)

        if stage in ("fit", "validate"):
            val_deferred_setup = (
                "validation_ds" in self._cfg
                and self._cfg.validation_ds is not None
                and self._cfg.validation_ds.get("defer_setup", False)
            )
            if self.val_dataloader() is None and val_deferred_setup:
                self.setup_multiple_validation_data(val_data_config=self._cfg.validation_ds)

        if stage == "test":
            test_deferred_setup = (
                "test_ds" in self._cfg
                and self._cfg.test_ds is not None
                and self._cfg.test_ds.get("defer_setup", False)
            )
            if self.test_dataloader() is None and test_deferred_setup:
                self.setup_multiple_test_data(test_data_config=self._cfg.test_ds)

    def train_dataloader(self):
        """Return the training dataloader."""
        return self._train_dl if self._train_dl is not None else None

    def val_dataloader(self):
        """Return the validation dataloader."""
        return self._validation_dl if self._validation_dl is not None else None

    def test_dataloader(self):
        """Return the test dataloader."""
        return self._test_dl if self._test_dl is not None else None

    def validation_epoch_end(
        self, outputs: Union[List[Dict[str, torch.Tensor]], List[List[Dict[str, torch.Tensor]]]]
    ) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
        """
        Default DataLoader for Validation set which automatically supports multiple data loaders
        via `multi_validation_epoch_end`.
        If multi dataset support is not required, override this method entirely in base class.
        In such a case, there is no need to implement `multi_validation_epoch_end` either.

        .. note::
            If more than one data loader exists, and they all provide `val_loss`,
            only the `val_loss` of the first data loader will be used by default.
            This default can be changed by passing the special key `val_dl_idx: int`
            inside the `validation_ds` config.

        Parameters
        ----------
        outputs: Single or nested list of tensor outputs from one or more data loaders.

        Returns
        -------
        A dictionary containing the union of all items from individual data_loaders, along with merged logs from all
        data loaders.
        """
        # Case where we dont provide data loaders
        if outputs is not None and len(outputs) == 0:
            return {}

        # Case where we provide exactly 1 data loader
        if type(outputs[0]) is dict:
            output_dict = self.multi_validation_epoch_end(outputs, dataloader_idx=0)  # type: ignore

            if output_dict is not None and "log" in output_dict:
                self.log_dict(output_dict.pop("log"), on_epoch=True)  # type: ignore

            return output_dict

        output_dict = {"log": {}}

        # The output is a list of list of dicts, outer list corresponds to dataloader idx
        for dataloader_idx, val_outputs in enumerate(outputs):  # type: ignore
            # Get prefix and dispatch call to multi epoch end
            dataloader_prefix = self.get_validation_dataloader_prefix(dataloader_idx)
            dataloader_logs = self.multi_validation_epoch_end(val_outputs, dataloader_idx=dataloader_idx)

            # If result was not provided, generate empty dict
            dataloader_logs: Dict[Any, Any] = dataloader_logs or {}  # type: ignore

            # Perform `val_loss` resolution first (if provided outside logs)
            if ("val_loss" in dataloader_logs and "val_loss" not in output_dict) and (  # type: ignore
                dataloader_idx == self._val_dl_idx
            ):
                output_dict["val_loss"] = dataloader_logs["val_loss"]  # type: ignore

            # For every item in the result dictionary
            for k, v in dataloader_logs.items():  # type: ignore
                # If the key is `log`
                if k == "log":
                    # Parse every element of the log, and attach the prefix name of the data loader
                    log_dict = {}

                    for k_log, v_log in v.items():
                        # If we are logging the metric, but dont provide it at result level,
                        # store it twice - once in log and once in result level.
                        # Also mark log with prefix name to avoid log level clash with other data loaders
                        if k_log not in output_dict["log"] and dataloader_idx == self._val_dl_idx:  # type: ignore
                            new_k_log = k_log

                            # Also insert duplicate key with prefix for ease of comparison / avoid name clash
                            log_dict[dataloader_prefix + k_log] = v_log

                        else:
                            # Simply prepend prefix to key and save
                            new_k_log = dataloader_prefix + k_log

                        # Store log value
                        log_dict[new_k_log] = v_log

                    # Update log storage of individual data loader
                    output_logs = output_dict["log"]  # type: ignore
                    output_logs.update(log_dict)

                    # Update global log storage
                    output_dict["log"] = output_logs  # type: ignore

                else:
                    # If any values are stored outside 'log', simply prefix name and store
                    new_k = dataloader_prefix + k
                    output_dict[new_k] = v  # type: ignore

        if "log" in output_dict:  # type: ignore
            self.log_dict(output_dict.pop("log"), on_epoch=True)  # type: ignore

        # return everything else
        return output_dict

    def test_epoch_end(
        self, outputs: Union[List[Dict[str, torch.Tensor]], List[List[Dict[str, torch.Tensor]]]]
    ) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
        """
        Default DataLoader for Test set which automatically supports multiple data loaders
        via `multi_test_epoch_end`.
        If multi dataset support is not required, override this method entirely in base class.
        In such a case, there is no need to implement `multi_test_epoch_end` either.

        .. note::
            If more than one data loader exists, and they all provide `test_loss`,
            only the `test_loss` of the first data loader will be used by default.
            This default can be changed by passing the special key `_test_dl_idx: int`
            inside the `test_ds` config.

        Parameters
        ----------
        outputs: Single or nested list of tensor outputs from one or more data loaders.

        Returns
        -------
        A dictionary containing the union of all items from individual data_loaders, along with merged logs from all
        data loaders.
        """
        # Case where we dont provide data loaders
        if outputs is not None and len(outputs) == 0:
            return {}

        # Case where we provide exactly 1 data loader
        if type(outputs[0]) is dict:
            output_dict = self.multi_test_epoch_end(outputs, dataloader_idx=0)  # type: ignore

            if output_dict is not None and "log" in output_dict:
                self.log_dict(output_dict.pop("log"), on_epoch=True)  # type: ignore

            return output_dict

        output_dict = {"log": {}}

        # The output is a list of dicts, outer list corresponds to dataloader idx
        for dataloader_idx, test_outputs in enumerate(outputs):  # type: ignore
            # Get prefix and dispatch call to multi epoch end
            dataloader_prefix = self.get_test_dataloader_prefix(dataloader_idx)
            self.multi_test_epoch_end(test_outputs, dataloader_idx=dataloader_idx)

            # If result was not provided, generate empty dict
            dataloader_logs = dataloader_logs or {}  # type: ignore

            # Perform `test_loss` resolution first (if provided outside logs)
            if (
                "test_loss" in dataloader_logs
                and "test_loss" not in output_dict  # type: ignore
                and dataloader_idx == self._test_dl_idx
            ):
                output_dict["test_loss"] = dataloader_logs["test_loss"]  # type: ignore

            # For every item in the result dictionary
            for k, v in dataloader_logs.items():
                # If the key is `log`
                if k == "log":
                    # Parse every element of the log, and attach the prefix name of the data loader
                    log_dict = {}
                    for k_log, v_log in v.items():
                        # If we are logging the loss, but dont provide it at result level,
                        # store it twice - once in log and once in result level.
                        # Also mark log with prefix name to avoid log level clash with other data loaders
                        if k_log not in output_dict["log"] and dataloader_idx == self._test_dl_idx:  # type: ignore
                            new_k_log = k_log

                            # Also insert duplicate key with prefix for ease of comparison / avoid name clash
                            log_dict[dataloader_prefix + k_log] = v_log

                        else:
                            # Simply prepend prefix to key and save
                            new_k_log = dataloader_prefix + k_log

                        log_dict[new_k_log] = v_log

                    # Update log storage of individual data loader
                    output_logs = output_dict.get("log", {})  # type: ignore
                    output_logs.update(log_dict)

                    # Update global log storage
                    output_dict["log"] = output_logs  # type: ignore

                else:
                    # If any values are stored outside 'log', simply prefix name and store
                    new_k = dataloader_prefix + k
                    output_dict[new_k] = v  # type: ignore

        if "log" in output_dict:  # type: ignore
            self.log_dict(output_dict.pop("log"), on_epoch=True)  # type: ignore

        # return everything else
        return output_dict

    @staticmethod
    def multi_validation_epoch_end(
        outputs: Union[object, List[Dict[str, torch.Tensor]], None], dataloader_idx: int = 0
    ) -> None:
        """
        Adds support for multiple validation datasets. Should be overridden by subclass, to obtain appropriate logs for
         each of the dataloaders.

        Parameters
        ----------
        outputs: Same as that provided by LightningModule.validation_epoch_end() for a single dataloader.
        dataloader_idx: int representing the index of the dataloader.

        Returns
        -------
        A dictionary of values, optionally containing a sub-dict `log`, such that the values in the log will be
        pre-pended by the dataloader prefix.
        """
        logging.warning(
            "Multi data loader support has been enabled, but `multi_validation_epoch_end(outputs, dataloader_idx) "
            "has not been implemented.\n"
            "If you require multi data loader support for validation sets, please override this method.\n"
            "If you do not require multi data loader support, please instead override `validation_epoch_end(outputs)."
        )

    @staticmethod
    def multi_test_epoch_end(outputs: Union[object, List[Dict[str, torch.Tensor]]], dataloader_idx: int = 0) -> None:
        """
        Adds support for multiple test datasets. Should be overridden by subclass, to obtain appropriate logs for each
        of the dataloaders.

        Parameters
        ----------
        outputs: Same as that provided by LightningModule.validation_epoch_end() for a single dataloader.
        dataloader_idx: int representing the index of the dataloader.

        Returns
        -------
        A dictionary of values, optionally containing a sub-dict `log`, such that the values in the log will be
        pre-pended by the dataloader prefix.
        """
        logging.warning(
            "Multi data loader support has been enabled, but `multi_test_epoch_end(outputs, dataloader_idx) has not "
            "been implemented.\n"
            "If you require multi data loader support for validation sets, please override this method.\n"
            "If you do not require multi data loader support, please instead override test_epoch_end(outputs)."
        )

    def get_validation_dataloader_prefix(self, dataloader_idx: int = 0) -> str:
        """Get the name of one or more data loaders, which will be prepended to all logs."""
        return self.validation_names[dataloader_idx]  # type: ignore

    def get_test_dataloader_prefix(self, dataloader_idx: int = 0) -> str:
        """Get the name of one or more data loaders, which will be prepended to all logs."""
        return self.test_names[dataloader_idx]  # type: ignore

    def load_part_of_state_dict(self, state_dict, include, exclude, load_from_string=None):
        """Load part of the state dict."""
        excluded_param_names = []
        # create dict
        dict_to_load = {}
        for k, v in state_dict.items():
            should_add = any(p in k for p in include)
            # except for if any string from exclude is present
            for e in exclude:
                if e in k:
                    excluded_param_names.append(k)
                    should_add = False
                    break
            if should_add:
                dict_to_load[k] = v

        # Restore checkpoint part into current model
        self.load_state_dict(dict_to_load, strict=False)  # type: ignore
        if load_from_string is not None:
            logging.info(f"Model checkpoint partially restored from {load_from_string}")
            if len(excluded_param_names) > 0:
                logging.info(
                    f"The following parameters were excluded when loading from {load_from_string} : "
                    f"{excluded_param_names}"
                )
                logging.info("Make sure that this is what you wanted!")
        else:
            if len(excluded_param_names) > 0:
                logging.info(
                    f"The following parameters were excluded when loading checkpoint : {excluded_param_names}"
                )

    @rank_zero_only
    def maybe_init_from_pretrained_checkpoint(self, cfg: OmegaConf, map_location: str = "cpu"):
        """
        Initializes a given model with the parameters obtained via specific config arguments. The state dict of the \
        provided model will be updated with `strict=False` setting to prevent requirement of exact model parameters \
        matching.

        Initializations

        init_from_mridc_model: Str path to a .mridc model, which will be instantiated in order to extract the state \
        dict.

        init_from_pretrained_model: Str name of a pretrained model checkpoint (obtained via cloud). The model will \
        be downloaded (or a cached copy will be used), instantiated and then its state dict will be extracted.

        init_from_ptl_ckpt: Str name of a Pytorch Lightning checkpoint file. It will be loaded and the state dict \
        will extract.

        Parameters
        ----------
        cfg: The config used to instantiate the model. It needs only contain one of the above keys.
        map_location: str or torch.device() which represents where the intermediate state dict (from the pretrained \
        model or checkpoint) will be loaded.
        """
        args = ["init_from_mridc_model", "init_from_pretrained_model", "init_from_ptl_ckpt"]
        arg_matches = [(1 if arg in cfg and arg is not None else 0) for arg in args]

        if sum(arg_matches) == 0:
            # model weights do not need to be restored
            return

        if sum(arg_matches) > 1:
            raise ValueError(
                "Cannot pass more than one model initialization arguments to config!\n"
                f"Found : {[args[idx] for idx, arg_present in enumerate(arg_matches) if arg_present]}"
            )

        if "init_from_mridc_model" in cfg and cfg.init_from_mridc_model is not None:  # type: ignore
            with open_dict(cfg):  # type: ignore
                if isinstance(cfg.init_from_mridc_model, str):  # type: ignore
                    model_path = cfg.init_from_mridc_model  # type: ignore
                    # Restore model
                    restored_model = self.restore_from(
                        model_path, map_location=map_location, strict=cfg.get("init_strict", True)  # type: ignore
                    )
                    # Restore checkpoint into current model
                    self.load_state_dict(restored_model.state_dict(), strict=False)
                    logging.info(f"Model checkpoint restored from mridc file with path : `{model_path}`")
                elif isinstance(cfg.init_from_mridc_model, (DictConfig, dict)):  # type: ignore
                    model_load_dict = cfg.init_from_mridc_model  # type: ignore
                    for model_load_cfg in model_load_dict.values():
                        model_path = model_load_cfg.path
                        # Restore model
                        restored_model = self.restore_from(
                            model_path, map_location=map_location, strict=cfg.get("init_strict", True)  # type: ignore
                        )

                        include = model_load_cfg.pop("include", [""])
                        exclude = model_load_cfg.pop("exclude", [])

                        self.load_part_of_state_dict(
                            restored_model.state_dict(), include, exclude, f"mridc file with path `{model_path}`"
                        )
                else:
                    raise TypeError("Invalid type: init_from_mridc_model is not a string or a dict!")

        if "init_from_pretrained_model" in cfg and cfg.init_from_pretrained_model is not None:  # type: ignore
            with open_dict(cfg):  # type: ignore
                # Restore model
                if isinstance(cfg.init_from_pretrained_model, str):  # type: ignore
                    model_name = cfg.pop("init_from_pretrained_model")  # type: ignore

                    # Check if model is being resumed or not - only works if `Trainer` is attached to model
                    if hasattr(self, "trainer") and self.trainer is not None:
                        trainer = self.trainer
                        if (
                            hasattr(trainer, "resume_from_checkpoint")
                            and trainer.checkpoint_connector.resume_checkpoint_path is not None
                        ):
                            logging.info(
                                "Model training is being resumed via Pytorch Lightning.\n"
                                "Initialization from pretrained model (via cloud) will be skipped."
                            )
                            return

                    restored_model = self.from_pretrained(
                        model_name, map_location=map_location, strict=cfg.get("init_strict", True)  # type: ignore
                    )

                    # Restore checkpoint into current model
                    self.load_state_dict(restored_model.state_dict(), strict=False)
                    logging.info(f"Model checkpoint restored from pretrained checkpoint with name : `{model_name}`")
                elif isinstance(cfg.init_from_pretrained_model, dict):  # type: ignore
                    pass
                elif isinstance(cfg.init_from_pretrained_model, (DictConfig, dict)):  # type: ignore
                    model_load_dict = cfg.init_from_pretrained_model  # type: ignore
                    for model_load_cfg in model_load_dict.values():
                        model_name = model_load_cfg.name
                        # Restore model
                        restored_model = self.from_pretrained(
                            model_name, map_location=map_location, strict=cfg.get("init_strict", True)  # type: ignore
                        )

                        include = model_load_cfg.pop("include", [""])
                        exclude = model_load_cfg.pop("exclude", [])

                        self.load_part_of_state_dict(
                            restored_model.state_dict(),
                            include,
                            exclude,
                            f"pretrained checkpoint with name `{model_name}`",
                        )
                else:
                    raise TypeError("Invalid type: init_from_pretrained_model is not a string or a dict!")

        if "init_from_ptl_ckpt" in cfg and cfg.init_from_ptl_ckpt is not None:  # type: ignore
            with open_dict(cfg):  # type: ignore
                if isinstance(cfg.init_from_ptl_ckpt, str):  # type: ignore
                    # Restore checkpoint
                    ckpt_path = cfg.pop("init_from_ptl_ckpt")  # type: ignore
                    ckpt = torch.load(ckpt_path, map_location=map_location)

                    # Restore checkpoint into current model
                    self.load_state_dict(ckpt["state_dict"], strict=False)
                    logging.info(
                        f"Model checkpoint restored from pytorch lightning checkpoint with path : `{ckpt_path}`"
                    )
                elif isinstance(cfg.init_from_ptl_ckpt, (DictConfig, dict)):  # type: ignore
                    model_load_dict = cfg.init_from_ptl_ckpt  # type: ignore
                    for model_load_cfg in model_load_dict.values():
                        ckpt_path = model_load_cfg.path
                        # Restore model
                        ckpt = torch.load(ckpt_path, map_location=map_location)

                        include = model_load_cfg.pop("include", [""])
                        exclude = model_load_cfg.pop("exclude", [])

                        self.load_part_of_state_dict(
                            ckpt["state_dict"], include, exclude, f"mridc file with path `{ckpt_path}`"
                        )
                else:
                    raise TypeError("Invalid type: init_from_ptl_ckpt is not a string or a dict!")

    def teardown(self, stage: str):
        """Called at the end of fit and test."""
        if stage == "fit" and "PL_TRAINER_GPUS" in os.environ:
            os.environ.pop("PL_TRAINER_GPUS")

        super().teardown(stage)

    @classmethod
    def extract_state_dict_from(
        cls,
        restore_path: str,
        save_dir: str,
        split_by_module: bool = False,
        save_restore_connector: SaveRestoreConnector = None,
    ):
        """
        Extract the state dict(s) from a provided .mridc tarfile and save it to a directory.

        Parameters
        ----------
        restore_path: path to .mridc file from which state dict(s) should be extracted
        save_dir: directory in which the saved state dict(s) should be stored
        split_by_module: bool flag, which determines whether the output checkpoint should be for the entire Model, or
        the individual module's that comprise the Model
        save_restore_connector: Can be overridden to add custom save and restore logic.

        Example
        -------
        To convert the .mridc tarfile into a single Model level PyTorch checkpoint

        .. code-block::

            state_dict = mridc.collections.asr.models.EncDecCTCModel.extract_state_dict_from('asr.mridc', \
            './asr_ckpts')

        To restore a model from a Model level checkpoint

        .. code-block::

            model = mridc.collections.asr.models.EncDecCTCModel(cfg)  # or any other method of restoration
            model.load_state_dict(torch.load("./asr_ckpts/model_weights.ckpt"))

        To convert the .mridc tarfile into multiple Module level PyTorch checkpoints

        .. code-block::

            state_dict = mridc.collections.asr.models.EncDecCTCModel.extract_state_dict_from('asr.mridc', \
            './asr_ckpts', split_by_module=True)

        To restore a module from a Module level checkpoint

        .. code-block::

            model = mridc.collections.asr.models.EncDecCTCModel(cfg)  # or any other method of restoration
            # load the individual components
            model.preprocessor.load_state_dict(torch.load("./asr_ckpts/preprocessor.ckpt"))
            model.encoder.load_state_dict(torch.load("./asr_ckpts/encoder.ckpt"))
            model.decoder.load_state_dict(torch.load("./asr_ckpts/decoder.ckpt"))

        Returns
        -------
        The state dict that was loaded from the original .mridc checkpoint.
        """
        if save_restore_connector is None:
            save_restore_connector = SaveRestoreConnector()

        if not path.exists(restore_path):
            raise FileExistsError(f"Can't find {restore_path}")

        cls.update_save_restore_connector(save_restore_connector)
        return cls._save_restore_connector.extract_state_dict_from(restore_path, save_dir, split_by_module)

    def prepare_test(self, trainer: "Trainer") -> bool:
        """
        Helper method to check whether the model can safely be tested on a dataset after training (or loading a
        checkpoint).

        .. code-block::

            trainer = Trainer()
            if model.prepare_test(trainer):
                trainer.test(model)

        Returns
        -------
        Bool which declares the model safe to test. Provides warnings if it has to return False to guide the user.
        """
        if not hasattr(self._cfg, "test_ds"):
            logging.info("No `test_ds` config found within the manifest.")
            return False

        if trainer is not None and trainer.num_devices > 1:
            # Replace ddp multi-gpu until PTL has a fix
            DDP_WARN = """\n\nDuring testing, it is currently advisable to construct a new Trainer "
                    "with single GPU and no DDP to obtain accurate results.
                    "Following pattern should be used: "
                    "trainer = Trainer(devices=1, accelerator='gpu')
                    "if model.prepare_test(trainer):"
                    "  trainer.test(model)\n\n"""

            logging.warning(DDP_WARN)
            return False

        # Assign trainer to the model
        self.set_trainer(trainer)
        return True

    def set_trainer(self, trainer: Trainer):
        """Set an instance of Trainer object."""
        self.trainer = trainer
        self._trainer = trainer
        self.set_world_size(self._trainer)

    def set_world_size(self, trainer: Trainer):
        """Determines the world size from the PyTorch Lightning Trainer and then updates AppState."""
        self.world_size = 1

        if trainer is not None:
            if isinstance(trainer, Trainer):
                if trainer.num_devices and trainer.num_nodes:
                    self.world_size = trainer.num_devices * trainer.num_nodes
            else:
                logging.warning("World size can only be set by PyTorch Lightning Trainer.")
        app_state = AppState()
        app_state.world_size = self.world_size

    def summarize(self, max_depth: int = 1) -> model_summary.ModelSummary:
        """Summarize this LightningModule."""
        return model_summary.summarize(self, max_depth=max_depth)

    def _update_dataset_config(self, dataset_name: str, config: Optional[Union[DictConfig, Dict]]):
        """
        Update the config (if not None) of the dataset by given name. Preserves said config after updating.

        Parameters
        ----------
        dataset_name: str name of the dataset whose config is being updated. Can be one of `train`, `validation` and
        `test`.
        config: Optional DictConfig or dict. If None is passed, this method simply returns. If dict is passed, it is
        cast into a DictConfig. The internal config is updated with the passed config.
        """
        if hasattr(self, "_multi_dataset_mode") and self._multi_dataset_mode is True:
            return

        if config is not None:
            if not isinstance(config, DictConfig):
                config = OmegaConf.create(config)

            if dataset_name in {"train", "validation", "test"}:
                OmegaConf.set_struct(self.cfg, False)

                key_name = f"{dataset_name}_ds"
                self.cfg[key_name] = config

                OmegaConf.set_struct(self.cfg, True)

                # Update hyperparameters by calling property setter
                self.cfg = self._cfg
            else:
                raise ValueError("`dataset_name` when updating config must be one of [train, validation, test]")

    @property
    def num_weights(self):
        """Utility property that returns the total number of parameters of the Model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def cfg(self):
        """
        Property that holds the finalized internal config of the model.

        .. note::
            Changes to this config are not reflected in the state of the model.
            Please create a new model using an updated config to properly update the model.
        """
        return self._cfg

    @cfg.setter
    def cfg(self, cfg):
        """
        Property that holds the finalized internal config of the model.

        .. note::
            Changes to this config are not reflected in the state of the model.
            Please create a new model using an updated config to properly update the model.
        """
        self._cfg = cfg
        self._set_hparams(OmegaConf.create({"cfg": self._cfg}))

        # TODO: Remove this when we have a better way to handle this
        if hasattr(self, "_hparams_initial") and "cfg" in self._hparams_initial:
            self._hparams_initial["cfg"] = OmegaConf.to_object(self._cfg)

    @staticmethod
    def _is_model_being_restored() -> bool:
        """Checks if the model is being restored from a checkpoint."""
        app_state = AppState()
        return app_state.is_model_being_restored

    @staticmethod
    def _set_model_restore_state(is_being_restored: bool, folder: str = None):
        """Sets the state of the model to be restored."""
        app_state = AppState()
        app_state.is_model_being_restored = is_being_restored
        app_state.mridc_file_folder = folder  # type: ignore

    def _set_model_guid(self):
        """Sets the model guid."""
        if not hasattr(self, "model_guid"):
            appstate = AppState()

            # Generate a unique uuid for the instance
            # also determine if the model is being restored or not, and preserve the path
            self.model_guid = str(uuid.uuid4())
            if self._is_model_being_restored():
                restore_path = appstate.model_restore_path
            else:
                restore_path = None

            appstate.register_model_guid(self.model_guid, restoration_path=restore_path)

    @classmethod
    def update_save_restore_connector(cls, save_restore_connector):
        """Update the save_restore_connector of the model."""
        if hasattr(cls, "_save_restore_connector"):
            cls._save_restore_connector = save_restore_connector
        else:
            setattr(cls, "_save_restore_connector", save_restore_connector)

    def _setup_nsys_profiling(self):
        """
        Enables nsys profiling To use, add the following options to the model config:
            nsys_profile: False
                start_step: 10  # Global batch to start profiling
                end_step: 10 # Global batch to end profiling
                ranks: [0] # Global rank IDs to profile
                gen_shape: False # Generate model and kernel details including input shapes

        And then wrap the model training script with:
            nsys profile -s none -o <profile filepath>  -t cuda,nvtx --force-overwrite true \
            --capture-range=cudaProfilerApi --capture-range-end=stop python ./examples/...
        See more options at: https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cli-profiling
        """
        if self.cfg.get("nsys_profile", None) is not None:
            if self.cfg.nsys_profile.get("enabled", False):
                # Nsys profiling options
                self._nsys_profile_enabled = True
                self._nsys_profile_start_step = self.cfg.nsys_profile.get("start_step", 0)
                self._nsys_profile_end_step = self.cfg.nsys_profile.get("end_step", 0)
                self._nsys_profile_ranks = self.cfg.nsys_profile.get("ranks", [0])
                self._nsys_profile_gen_shape = self.cfg.nsys_profile.get("gen_shape", False)

                if type(self._nsys_profile_start_step) == int:
                    logging.info(f"Nsys profiling setup with start_step: {self._nsys_profile_start_step}")
                else:
                    raise ValueError(
                        f"Nsys start_step must be of type int. Found: {type(self._nsys_profile_start_step)}"
                    )

                if type(self._nsys_profile_end_step) == int:
                    logging.info(f"Nsys profiling setup with end_step: {self._nsys_profile_end_step}")
                else:
                    raise ValueError(f"Nsys end_step must be of type int. Found: {type(self._nsys_profile_end_step)}")

                if self._nsys_profile_end_step >= self._nsys_profile_start_step:
                    pass
                else:
                    raise ValueError(f"Nsys end_step must be greater than or equal to nsys start_step")

    def on_train_batch_start(self, batch: Any, batch_idx: int, unused: int = 0):
        """PyTorch Lightning hook:
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-train-batch-start
        We use it here to enable nsys profiling.
        """
        if self.device.type == "cuda":
            if hasattr(self, "_nsys_profile_enabled"):
                if self._nsys_profile_enabled:
                    if batch_idx == self._nsys_profile_start_step and get_rank() in self._nsys_profile_ranks:
                        logging.info("====== Start nsys profiling ======")
                        torch.cuda.cudart().cudaProfilerStart()
                        if self._nsys_profile_gen_shape:
                            torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int, unused: int = 0) -> None:
        """PyTorch Lightning hook:
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-train-batch-end
        We use it here to enable nsys profiling.
        """
        if self.device.type == "cuda":
            if hasattr(self, "_nsys_profile_enabled"):
                if self._nsys_profile_enabled:
                    if batch_idx == self._nsys_profile_end_step and get_rank() in self._nsys_profile_ranks:
                        logging.info("====== End nsys profiling ======")
                        torch.cuda.cudart().cudaProfilerStop()
