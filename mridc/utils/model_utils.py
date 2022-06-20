# encoding: utf-8
import sys

__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/utils/model_utils.py
import copy
import os
from dataclasses import dataclass, is_dataclass
from packaging.version import Version
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional, Set, Tuple, Union

import wrapt
from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException
from pytorch_lightning import LightningModule

import mridc
from mridc.constants import MRIDC_ENV_CACHE_DIR
from mridc.core.classes.common import PretrainedModelInfo
from mridc.core.classes.modelPT import ModelPT
from mridc.core.conf.modelPT import MRIDCConfig
from mridc.utils.app_state import AppState
from mridc.utils import logging

_HAS_HYDRA = True

_VAL_TEST_FASTPATH_KEY = "ds_item"

__all__ = [
    "ArtifactPathType",
    "ArtifactItem",
    "resolve_dataset_name_from_cfg",
    "parse_dataset_as_name",
    "unique_names_check",
    "resolve_validation_dataloaders",
    "wrap_training_step",
    "convert_model_config_to_dict_config",
    "_convert_config",
    "maybe_update_config_version",
    "import_class_by_path",
    "resolve_subclass_pretrained_model_info",
    "check_lib_version",
    "resolve_cache_dir",
    "inject_model_parallel_rank",
    "uninject_model_parallel_rank",
]


class ArtifactPathType(Enum):
    """
    ArtifactPathType refers to the type of the path that the artifact is located at.
    LOCAL_PATH: A user local filepath that exists on the file system.
    TAR_PATH: A (generally flattened) filepath that exists inside of an archive (that may have its own full path).
    """

    LOCAL_PATH = 0
    TAR_PATH = 1


@dataclass(init=False)
class ArtifactItem:
    """ArtifactItem is a dataclass that holds the information of an artifact."""

    path: str
    path_type: ArtifactPathType
    hashed_path: Optional[str] = None


def resolve_dataset_name_from_cfg(cfg: "DictConfig") -> Union[Union[str, int, Enum, float, bool, None], Any]:
    """
    Parses items of the provided sub-config to find the first potential key that resolves to an existing file or
    directory.

    # Fast-path Resolution
    In order to handle cases where we need to resolve items that are not paths, a fastpath key can be provided as
    defined in the global `_VAL_TEST_FASTPATH_KEY`.

    This key can be used in two ways :
    ## _VAL_TEST_FASTPATH_KEY points to another key in the config
    If this _VAL_TEST_FASTPATH_KEY points to another key in this config itself, then we assume we want to loop through
    the values of that key. This allows for any key in the config to become a fastpath key.

    Example
    -------
    validation_ds:

    .. code-block::

        splits: "val"
        ...
        <_VAL_TEST_FASTPATH_KEY>: "splits"  <-- this points to the key name "splits"

    Then we can write the following when overriding in hydra:
    ```python
    python train_file.py ... model.validation_ds.splits=[val1, val2, dev1, dev2] ...
    ```
    ## _VAL_TEST_FASTPATH_KEY itself acts as the resolved key
    If this _VAL_TEST_FASTPATH_KEY does not point to another key in the config, then it is assumed that the items of
    this key itself are used for resolution.

    Example
    -------
    validation_ds:

    .. code-block::

        <_VAL_TEST_FASTPATH_KEY>: "val"  <-- this points to the key name "splits"

    Then we can write the following when overriding in hydra:
    ```python
    python train_file.py ... model.validation_ds.<_VAL_TEST_FASTPATH_KEY>=[val1, val2, dev1, dev2] ...
    ```
    # IMPORTANT NOTE:
    It <can> potentially mismatch if there exist more than 2 valid paths, and the first path does *not* resolve the
    path of the data file (but does resolve to some other valid path). To avoid this side effect, place the data path
    as the first item on the config file.

    Parameters
    ----------
    cfg: Sub-config of the config file.

    Returns
    -------
    A str representing the `key` of the config which hosts the filepath(s), or None in case path could not be resolved.
    """
    if _VAL_TEST_FASTPATH_KEY in cfg and cfg[_VAL_TEST_FASTPATH_KEY] is not None:
        fastpath_key = cfg[_VAL_TEST_FASTPATH_KEY]

        if isinstance(fastpath_key, str) and fastpath_key in cfg:
            return cfg[fastpath_key]
        return _VAL_TEST_FASTPATH_KEY

    for key, value in cfg.items():
        if type(value) in [list, tuple, ListConfig]:
            # Count the number of valid paths in the list
            values_are_paths = 0
            for val_i in value:
                val_i = str(val_i)

                if os.path.exists(val_i) or os.path.isdir(val_i):
                    values_are_paths += 1
                else:
                    # reset counter and break inner loop
                    break

            if values_are_paths == len(value):
                return key

        elif os.path.exists(str(value)) or os.path.isdir(str(value)):
            return key

    return None


def parse_dataset_as_name(name: str) -> str:
    """
    Constructs a valid prefix-name from a provided file path.

    Parameters
    ----------
    name: Path to some valid data/manifest file or a python object that will be used as a name for the data loader (via
    str() cast).

    Returns
    -------
    A valid prefix-name for the data loader.
    """
    name = Path(name).stem if os.path.exists(name) or os.path.isdir(name) else name
    # cleanup name
    name = name.replace("-", "_")

    if "manifest" in name:
        name = name.replace("manifest", "")

    if "dataset" in name:
        name = name.replace("dataset", "")

    # Test if the manifest/dataset name was simply `manifest.yaml` or `dataset.yaml`: Invalid names.
    if name == "":
        raise ValueError(
            "Provided dataset / manifest filename was `manifest.json` or `dataset.json`.\n"
            "Such a name is invalid, since multiple datasets/manifests can share the same name,\n"
            "thereby overriding their results during logging. Please pick a more descriptive filename \n"
            "for the provided dataset / manifest file."
        )

    if name[-1] != "_":
        name = f"{name}_"

    return name


def unique_names_check(name_list: Optional[List[str]]):
    """
    Performs a uniqueness check on the name list resolved, so that it can warn users about non-unique keys.

    Parameters
    ----------
    name_list: List of strings resolved for data loaders.
    """
    if name_list is None:
        return

    # Name uniqueness checks
    names = set()
    for name in name_list:
        if name in names:
            logging.warning(
                "Name resolution has found more than one data loader having the same name !\n"
                "In such cases, logs will nor be properly generated. "
                "Please rename the item to have unique names.\n"
                f"Resolved name : {name}"
            )
        else:
            names.add(name)  # we need just hash key check, value is just a placeholder


def resolve_validation_dataloaders(model: ModelPT):
    """
    Helper method that operates on the ModelPT class to automatically support multiple dataloaders for the validation
    set. It does so by first resolving the path to one/more data files via `resolve_dataset_name_from_cfg()`.
    If this resolution fails, it assumes the data loader is prepared to manually support / not support multiple data
    loaders and simply calls the appropriate setup method.
    If resolution succeeds:
    - Checks if provided path is to a single file or a list of files.
    If a single file is provided, simply tags that file as such and loads it via the setup method.
    If multiple files are provided:
    - Inject a new manifest path at index "i" into the resolved key.
    - Calls the appropriate setup method to set the data loader.
    - Collects the initialized data loader in a list and preserves it.
    - Once all data loaders are processed, assigns the list of loaded loaders to the ModelPT.
    - Finally, assigns a list of unique names resolved from the file paths to the ModelPT.

    Parameters
    ----------
    model: ModelPT subclass, which requires >=1 Validation Dataloaders to be setup.
    """
    if not _HAS_HYDRA:
        logging.error("This function requires Hydra/OmegaConf and it was not installed.")
        sys.exit(1)
    cfg = copy.deepcopy(model._cfg)
    dataloaders: List[Any] = []

    # process val_loss_idx
    if "val_dl_idx" in cfg.validation_ds:
        cfg = OmegaConf.to_container(cfg)
        val_dl_idx = cfg["validation_ds"].pop("val_dl_idx")
        cfg = OmegaConf.create(cfg)
    else:
        val_dl_idx = 0

    # Set val_loss_idx
    model._val_dl_idx = val_dl_idx

    ds_key = resolve_dataset_name_from_cfg(cfg.validation_ds)

    if ds_key is None:
        logging.debug(
            f"Could not resolve file path from provided config - {cfg.validation_ds}. "
            "Disabling support for multi-dataloaders."
        )

        model.setup_validation_data(cfg.validation_ds)
        return

    ds_values = cfg.validation_ds[ds_key]

    if isinstance(ds_values, (list, tuple, ListConfig)):

        for ds_value in ds_values:
            cfg.validation_ds[ds_key] = ds_value
            model.setup_validation_data(cfg.validation_ds)
            dataloaders.append(model.validation_dl)

        model.validation_dl = dataloaders  # type: ignore
        model.validation_names = [parse_dataset_as_name(ds) for ds in ds_values]  # type: ignore

        unique_names_check(name_list=model.validation_names)
        return
    model.setup_validation_data(cfg.validation_ds)
    model.validation_names = [parse_dataset_as_name(ds_values)]

    unique_names_check(name_list=model.validation_names)


def resolve_test_dataloaders(model: "ModelPT"):
    """
    Helper method that operates on the ModelPT class to automatically support
    multiple dataloaders for the test set.
    It does so by first resolving the path to one/more data files via `resolve_dataset_name_from_cfg()`.
    If this resolution fails, it assumes the data loader is prepared to manually support / not support
    multiple data loaders and simply calls the appropriate setup method.
    If resolution succeeds:
        Checks if provided path is to a single file or a list of files.
        If a single file is provided, simply tags that file as such and loads it via the setup method.
        If multiple files are provided:
            Inject a new manifest path at index "i" into the resolved key.
            Calls the appropriate setup method to set the data loader.
            Collects the initialized data loader in a list and preserves it.
            Once all data loaders are processed, assigns the list of loaded loaders to the ModelPT.
            Finally, assigns a list of unique names resolved from the file paths to the ModelPT.

    Parameters
    ----------
    model: ModelPT subclass, which requires >=1 Test Dataloaders to be setup.
    """
    if not _HAS_HYDRA:
        logging.error("This function requires Hydra/OmegaConf and it was not installed.")
        sys.exit(1)
    cfg = copy.deepcopy(model._cfg)
    dataloaders: List[Any] = []

    # process test_loss_idx
    if "test_dl_idx" in cfg.test_ds:
        cfg = OmegaConf.to_container(cfg)
        test_dl_idx = cfg["test_ds"].pop("test_dl_idx")
        cfg = OmegaConf.create(cfg)
    else:
        test_dl_idx = 0

    # Set val_loss_idx
    model._test_dl_idx = test_dl_idx

    ds_key = resolve_dataset_name_from_cfg(cfg.test_ds)

    if ds_key is None:
        logging.debug(
            f"Could not resolve file path from provided config - {cfg.test_ds}. "
            "Disabling support for multi-dataloaders."
        )

        model.setup_test_data(cfg.test_ds)
        return

    ds_values = cfg.test_ds[ds_key]

    if isinstance(ds_values, (list, tuple, ListConfig)):

        for ds_value in ds_values:
            cfg.test_ds[ds_key] = ds_value
            model.setup_test_data(cfg.test_ds)
            dataloaders.append(model.test_dl)

        model.test_dl = dataloaders  # type: ignore
        model.test_names = [parse_dataset_as_name(ds) for ds in ds_values]  # type: ignore

        unique_names_check(name_list=model.test_names)
        return
    model.setup_test_data(cfg.test_ds)
    model.test_names = [parse_dataset_as_name(ds_values)]

    unique_names_check(name_list=model.test_names)


@wrapt.decorator
def wrap_training_step(wrapped, instance: LightningModule, args, kwargs):
    """
    Wraps the training step of the LightningModule.

    Parameters
    ----------
    wrapped: The wrapped function.
    instance: The LightningModule instance.
    args: The arguments passed to the wrapped function.
    kwargs: The keyword arguments passed to the wrapped function.

    Returns
    -------
    The return value of the wrapped function.
    """
    output_dict = wrapped(*args, **kwargs)

    if isinstance(output_dict, dict) and output_dict is not None and "log" in output_dict:
        log_dict = output_dict.pop("log")
        instance.log_dict(log_dict, on_step=True)

    return output_dict


def convert_model_config_to_dict_config(cfg: Union[DictConfig, MRIDCConfig]) -> DictConfig:
    """
    Converts its input into a standard DictConfig.

    Possible input values are:
        - DictConfig
        - A dataclass which is a subclass of MRIDCConfig

    Parameters
    ----------
    cfg: A dict-like object.

    Returns
    -------
    The equivalent DictConfig.
    """
    if not _HAS_HYDRA:
        logging.error("This function requires Hydra/OmegaConf and it was not installed.")
        sys.exit(1)
    if not isinstance(cfg, (OmegaConf, DictConfig)) and is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)

    if not isinstance(cfg, DictConfig):
        raise ValueError(f"cfg constructor argument must be of type DictConfig/dict but got {type(cfg)} instead.")

    config = OmegaConf.to_container(cfg, resolve=True)
    config = OmegaConf.create(config)
    return config


def _convert_config(cfg: "OmegaConf"):
    """Recursive function converting the configuration from old hydra format to the new one."""
    if not _HAS_HYDRA:
        logging.error("This function requires Hydra/OmegaConf and it was not installed.")
        sys.exit(1)

    # Get rid of cls -> _target_.
    if "cls" in cfg and "_target_" not in cfg:
        cfg._target_ = cfg.pop("cls")  # type: ignore

    # Get rid of params.
    if "params" in cfg:
        params = cfg.pop("params")  # type: ignore
        for param_key, param_val in params.items():
            cfg[param_key] = param_val

    # Recursion.
    try:
        for _, sub_cfg in cfg.items():  # type: ignore
            if isinstance(sub_cfg, DictConfig):
                _convert_config(sub_cfg)  # type: ignore
    except OmegaConfBaseException as e:
        logging.warning(f"Skipped conversion for config/subconfig:\n{cfg}\n Reason: {e}.")


def maybe_update_config_version(cfg: "DictConfig"):
    """
    Recursively convert Hydra 0.x configs to Hydra 1.x configs.
    Changes include:
    -   `cls` -> `_target_`.
    -   `params` -> drop params and shift all arguments to parent.
    -   `target` -> `_target_` cannot be performed due to ModelPT injecting `target` inside class.

    Parameters
    ----------
    cfg: Any Hydra compatible DictConfig

    Returns
    -------
    An updated DictConfig that conforms to Hydra 1.x format.
    """
    if not _HAS_HYDRA:
        logging.error("This function requires Hydra/OmegaConf and it was not installed.")
        sys.exit(1)
    if cfg is not None and not isinstance(cfg, DictConfig):
        try:
            temp_cfg = OmegaConf.create(cfg)
            cfg = temp_cfg
        except OmegaConfBaseException:
            # Cannot be cast to DictConfig, skip updating.
            return cfg

    # Make a copy of model config.
    cfg = copy.deepcopy(cfg)
    OmegaConf.set_struct(cfg, False)

    # Convert config.
    _convert_config(cfg)  # type: ignore

    # Update model config.
    OmegaConf.set_struct(cfg, True)

    return cfg


def import_class_by_path(path: str):
    """Recursive import of class by path string."""
    paths = path.split(".")
    path = ".".join(paths[:-1])
    class_name = paths[-1]
    mod = __import__(path, fromlist=[class_name])
    mod = getattr(mod, class_name)
    return mod


def resolve_subclass_pretrained_model_info(base_class) -> Union[List[PretrainedModelInfo], Set[Any]]:
    """
    Recursively traverses the inheritance graph of subclasses to extract all pretrained model info.
    First constructs a set of unique pretrained model info by performing DFS over the inheritance graph.
    All model info belonging to the same class is added together.

    Parameters
    ----------
    base_class: The root class, whose subclass graph will be traversed.

    Returns
    -------
    A list of unique pretrained model infos belonging to all the inherited subclasses of this baseclass.
    """
    list_of_models = set()

    def recursive_subclass_walk(cls):
        """
        Recursively traverses the inheritance graph of subclasses to extract all pretrained model info.

        Parameters
        ----------
        cls: The class to be traversed.

        Returns
        -------
        A list of unique pretrained model infos belonging to all the inherited subclasses of this baseclass.
        """
        for subclass in cls.__subclasses__():
            # step into its immediate subclass
            recursive_subclass_walk(subclass)

            subclass_models = subclass.list_available_models()

            if subclass_models is not None and len(subclass_models) > 0:
                # Inject subclass info into pretrained model info, if not already overridden by subclass.
                for model_info in subclass_models:
                    # If subclass manually injects class_, dont override.
                    if model_info.class_ is None:
                        model_info.class_ = subclass

                for model_info in subclass_models:
                    list_of_models.add(model_info)

    recursive_subclass_walk(base_class)
    list_of_models = list(sorted(list_of_models))  # type: ignore
    return list_of_models


def check_lib_version(lib_name: str, checked_version: str, operator) -> Tuple[Optional[bool], str]:
    """
    Checks if a library is installed, and if it is, checks the operator(lib.__version__, checked_version) as a result.
    This bool result along with a string analysis of result is returned.
    If the library is not installed at all, then returns None instead, along with a string explaining
    that the library is not installed

    Parameters
    ----------
    lib_name: lower case str name of the library that must be imported.
    checked_version: semver string that is compared against lib.__version__.
    operator: binary callable function func(a, b) -> bool; that compares lib.__version__ against version in some
    manner. Must return a boolean.

    Returns
    -------
    A tuple of results:
        -   Bool or None. Bool if the library could be imported, and the result of
            operator(lib.__version__, checked_version) or False if __version__ is not implemented in lib.
            None is passed if the library is not installed at all.
        -   A string analysis of the check.
    """
    try:
        if "." in lib_name:
            mod = import_class_by_path(lib_name)
        else:
            mod = __import__(lib_name)

        if hasattr(mod, "__version__"):
            lib_ver = Version(mod.__version__)  # type: ignore
            match_ver = Version(checked_version)  # type: ignore

            if operator(lib_ver, match_ver):
                msg = f"Lib {lib_name} version is satisfied !"
                return True, msg
            msg = (
                f"Lib {lib_name} version ({lib_ver}) is not {operator.__name__} than required version "
                f"{checked_version}.\n"
                "Please upgrade the lib using either pip or conda to the latest version."
            )
            return False, msg
        msg = (
            f"Lib {lib_name} does not implement __version__ in its init file. "
            "Could not check version compatibility."
        )
        return False, msg
    except ImportError:
        pass

    msg = f"Lib {lib_name} has not been installed. Please use pip or conda to install this package."
    return None, msg


def resolve_cache_dir() -> Path:
    """
    Utility method to resolve a cache directory for MRIDC that can be overridden by an environment variable.
    Example:
        MRIDC_CACHE_DIR="~/mridc_cache_dir/" python mridc_example_script.py

    Returns
    -------
    A Path object, resolved to the absolute path of the cache directory. If no override is provided, uses an inbuilt
    default which adapts to mridc versions strings.
    """
    override_dir = os.environ.get(MRIDC_ENV_CACHE_DIR, "")
    return (
        Path.joinpath(Path.home(), f".cache/torch/MRIDC/MRIDC_{mridc.__version__}")
        if override_dir == ""
        else Path(override_dir).resolve()
    )


def uninject_model_parallel_rank(filepath):
    """Uninjects tensor/pipeline model parallel ranks from the filepath."""
    filepath = str(filepath)
    if "mp_rank" in filepath or "tp_rank" in filepath:
        dirname = os.path.dirname(os.path.dirname(filepath))
        basename = os.path.basename(filepath)
        filepath = os.path.join(dirname, basename)
    return filepath


def inject_model_parallel_rank(filepath):
    """Injects tensor/pipeline model parallel ranks into the filepath. Does nothing if not using model parallelism."""
    filepath = uninject_model_parallel_rank(filepath)
    app_state = AppState()
    if app_state.model_parallel_size is not None and app_state.model_parallel_size > 1:
        # filepath needs to be updated to include mp_rank
        dirname = os.path.dirname(filepath)
        basename = os.path.basename(filepath)
        if app_state.pipeline_model_parallel_size is None or app_state.pipeline_model_parallel_size == 1:
            filepath = f"{dirname}/mp_rank_{app_state.tensor_model_parallel_rank:02d}/{basename}"
        else:
            filepath = (
                f"{dirname}/tp_rank_{app_state.tensor_model_parallel_rank:02d}_pp_rank_"
                f"{app_state.pipeline_model_parallel_rank:03d}/{basename} "
            )
        return filepath
    return filepath
