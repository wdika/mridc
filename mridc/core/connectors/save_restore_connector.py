# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/core/connectors/save_restore_connector.py

import os
import shutil
import tarfile
import tempfile
import uuid
from typing import Optional, Union

import torch
from omegaconf import DictConfig, OmegaConf
from omegaconf.omegaconf import open_dict
from pytorch_lightning.trainer.trainer import Trainer

import mridc.utils
from mridc.utils import logging
from mridc.utils.app_state import AppState
from mridc.utils.get_rank import is_global_rank_zero


class SaveRestoreConnector:
    """This class is used to save and restore the model state."""

    def __init__(self) -> None:
        self._model_config_yaml = "model_config.yaml"
        self._model_weights_ckpt = "model_weights.ckpt"
        self._model_extracted_dir = None

    def save_to(self, model, save_path: str):
        """
        Saves model instance (weights and configuration) into .mridc file.
        You can use "restore_from" method to fully restore instance from .mridc file.
        .mridc file is an archive (tar.gz) with the following:
        - model_config.yaml - model configuration in .yaml format. You can deserialize this into cfg argument for \
        model's constructor
        - model_wights.chpt - model checkpoint

        Parameters
        ----------
        model: ModelPT object to be saved.
        save_path: Path to .mridc file where model instance should be saved
        """
        if is_global_rank_zero():
            with tempfile.TemporaryDirectory() as tmpdir:
                config_yaml = os.path.join(tmpdir, self.model_config_yaml)
                model_weights = os.path.join(tmpdir, self.model_weights_ckpt)
                model.to_config_file(path2yaml_file=config_yaml)
                if hasattr(model, "artifacts") and model.artifacts is not None:
                    self._handle_artifacts(model, mridc_file_folder=tmpdir)
                    # We should not update self._cfg here - the model can still be in use
                    self._update_artifact_paths(model, path2yaml_file=config_yaml)
                self._save_state_dict_to_disk(model.state_dict(), model_weights)
                self._make_mridc_file_from_folder(filename=save_path, source_dir=tmpdir)
        else:
            return

    def load_config_and_state_dict(
        self,
        calling_cls,
        restore_path: str,
        override_config_path: Optional[Union[OmegaConf, str]] = None,
        map_location: Optional[torch.device] = None,
        strict: bool = True,
        return_config: bool = False,
        trainer: Trainer = None,
    ):
        """
        Restores model instance (weights and configuration) into .mridc file

        Parameters
        ----------
        calling_cls: Class of the model to be restored.
        restore_path: path to .mridc file from which model should be instantiated
        override_config_path: path to a yaml config that will override the internal config file or an
        OmegaConf/DictConfig object representing the model config.
        map_location: Optional torch.device() to map the instantiated model to a device. By default (None), it will
        select a GPU if available, falling back to CPU otherwise.
        strict: Passed to load_state_dict. By default, True.
        return_config: If set to true, will return just the underlying config of the restored model as an OmegaConf
        DictConfig object without instantiating the model.
        trainer: Optional trainer object to be used for model parallelism.

        Example
        -------
            ```
            model = mridc.collections.asr.models.EncDecCTCModel.restore_from('asr.mridc')
            assert isinstance(model, mridc.collections.asr.models.EncDecCTCModel)
            ```

        Returns
        -------
        An instance of type cls or its underlying config (if return_config is set).
        """
        # Get path where the command is executed - the artifacts will be "retrieved" there
        # (original .mridc behavior)
        cwd = os.getcwd()

        if map_location is None:
            if torch.cuda.is_available():
                map_location = torch.device("cuda")
            else:
                map_location = torch.device("cpu")

        app_state = AppState()
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                # Check if self.model_extracted_dir is set, and is a valid path
                if self.model_extracted_dir is not None and os.path.isdir(self.model_extracted_dir):
                    # Log that MRIDC will use the provided `model_extracted_dir`
                    logging.info(
                        "Restoration will occur within pre-extracted directory : " f"`{self.model_extracted_dir}`."
                    )
                    # Override `tmpdir` above with the pre-extracted `model_extracted_dir`
                    tmpdir = self.model_extracted_dir
                else:
                    # Extract the nemo file into the temporary directory
                    self._unpack_mridc_file(path2file=restore_path, out_folder=tmpdir)

                # Change current working directory to the temporary directory
                os.chdir(tmpdir)
                if override_config_path is None:
                    config_yaml = os.path.join(tmpdir, self.model_config_yaml)
                else:
                    # can be str path or OmegaConf / DictConfig object
                    config_yaml = override_config_path
                if not isinstance(config_yaml, (OmegaConf, DictConfig)):
                    conf = OmegaConf.load(config_yaml)
                else:
                    conf = config_yaml
                    if override_config_path is not None:
                        # Resolve the override config
                        conf = OmegaConf.to_container(conf, resolve=True)
                        conf = OmegaConf.create(conf)
                # If override is top level config, extract just `model` from it
                if "model" in conf:
                    conf = conf.model

                if return_config:
                    instance = conf
                    return instance
                if app_state.model_parallel_rank is not None and app_state.model_parallel_size > 1:
                    model_weights = self._inject_model_parallel_rank_for_ckpt(tmpdir, self.model_weights_ckpt)
                else:
                    model_weights = os.path.join(tmpdir, self.model_weights_ckpt)
                OmegaConf.set_struct(conf, True)
                os.chdir(cwd)
                # get the class
                calling_cls._set_model_restore_state(is_being_restored=True, folder=tmpdir)  # type: ignore
                instance = calling_cls.from_config_dict(config=conf, trainer=trainer)
                instance = instance.to(map_location)
                # add load_state_dict override
                if app_state.model_parallel_size is not None and app_state.model_parallel_size > 1:
                    model_weights = self._inject_model_parallel_rank_for_ckpt(tmpdir, self.model_weights_ckpt)
                instance.load_state_dict(
                    self._load_state_dict_from_disk(model_weights, map_location=map_location), strict=strict
                )
                logging.info(f"Model {instance.__class__.__name__} was successfully restored from {restore_path}.")
                instance._set_model_restore_state(is_being_restored=False)  # type: ignore
            finally:
                os.chdir(cwd)

        return instance

    @staticmethod
    def load_instance_with_state_dict(instance, state_dict, strict):
        """Loads the state dict into the instance."""
        instance.load_state_dict(state_dict, strict=strict)
        instance._set_model_restore_state(is_being_restored=False)  # type: ignore

    def restore_from(
        self,
        calling_cls,
        restore_path: str,
        override_config_path: Optional[Union[OmegaConf, str]] = None,
        map_location: Optional[torch.device] = None,
        strict: bool = True,
        return_config: bool = False,
        trainer: Trainer = None,
    ):
        """
        Restores model instance (weights and configuration) into .mridc file

        Parameters
        ----------
        calling_cls: The class of the model to be restored.
        restore_path: path to .mridc file from which model should be instantiated
        override_config_path: path to a yaml config that will override the internal config file or an
        OmegaConf/DictConfig object representing the model config.
        map_location: Optional torch.device() to map the instantiated model to a device. By default (None), it will
        select a GPU if available, falling back to CPU otherwise.
        strict: Passed to load_state_dict. By default, True.
        return_config: If set to true, will return just the underlying config of the restored model as an
        OmegaConf/DictConfig object without instantiating the model.
        trainer: Optional trainer object to be used for restoring the model.

        Returns
        -------
        An instance of type cls or its underlying config (if return_config is set).
        """
        # Get path where the command is executed - the artifacts will be "retrieved" there (original .mridc behavior)
        loaded_params = self.load_config_and_state_dict(
            calling_cls,
            restore_path,
            override_config_path,
            map_location,
            strict,
            return_config,
            trainer,
        )

        if not isinstance(loaded_params, tuple):
            return loaded_params

        _, instance, state_dict = loaded_params
        self.load_instance_with_state_dict(instance, state_dict, strict)
        logging.info(f"Model {instance.__class__.__name__} was successfully restored from {restore_path}.")
        return instance

    def extract_state_dict_from(self, restore_path: str, save_dir: str, split_by_module: bool = False):
        """
        Extract the state dict(s) from a provided .mridc tarfile and save it to a directory.

        Parameters
        ----------
        restore_path: path to .mridc file from which state dict(s) should be extracted
        save_dir: directory in which the saved state dict(s) should be stored
        split_by_module: bool flag, which determines whether the output checkpoint should be for the entire Model, or
        the individual module's that comprise the Model.

        Example
        -------
        To convert the .mridc tarfile into a single Model level PyTorch checkpoint
        ::
        state_dict = mridc.collections.asr.models.EncDecCTCModel.extract_state_dict_from('asr.mridc',
        './asr_ckpts')
        To restore a model from a Model level checkpoint
        ::
        model = mridc.collections.asr.models.EncDecCTCModel(cfg)  # or any other method of restoration
        model.load_state_dict(torch.load("./asr_ckpts/model_weights.ckpt"))
        To convert the .mridc tarfile into multiple Module level PyTorch checkpoints
        ::
        state_dict = mridc.collections.asr.models.EncDecCTCModel.extract_state_dict_from('asr.mridc',
        './asr_ckpts', split_by_module=True). To restore a module from a Module level checkpoint
        ::
        model = mridc.collections.asr.models.EncDecCTCModel(cfg)  # or any other method of restoration
        # load the individual components
        model.preprocessor.load_state_dict(torch.load("./asr_ckpts/preprocessor.ckpt"))
        model.encoder.load_state_dict(torch.load("./asr_ckpts/encoder.ckpt"))
        model.decoder.load_state_dict(torch.load("./asr_ckpts/decoder.ckpt"))

        Returns
        -------
        The state dict that was loaded from the original .mridc checkpoint.
        """
        cwd = os.getcwd()

        save_dir = os.path.abspath(save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                self._unpack_mridc_file(path2file=restore_path, out_folder=tmpdir)
                os.chdir(tmpdir)
                model_weights = os.path.join(tmpdir, self.model_weights_ckpt)
                state_dict = self._load_state_dict_from_disk(model_weights)

                if not split_by_module:
                    filepath = os.path.join(save_dir, self.model_weights_ckpt)
                    self._save_state_dict_to_disk(state_dict, filepath)

                else:
                    key_set = {key.split(".")[0] for key in state_dict.keys()}
                    for primary_key in key_set:
                        inner_keys = [key for key in state_dict.keys() if key.split(".")[0] == primary_key]
                        state_dict_subset = {
                            ".".join(inner_key.split(".")[1:]): state_dict[inner_key] for inner_key in inner_keys
                        }
                        filepath = os.path.join(save_dir, f"{primary_key}.ckpt")
                        self._save_state_dict_to_disk(state_dict_subset, filepath)

                logging.info(f"Checkpoints from {restore_path} were successfully extracted into {save_dir}.")
            finally:
                os.chdir(cwd)

        return state_dict

    @staticmethod
    def register_artifact(model, config_path: str, src: str, verify_src_exists: bool = True):
        """
        Register model artifacts with this function. These artifacts (files) will be included inside .mridc file
        when model.save_to("mymodel.mridc") is called.

        How it works:
        1. It always returns existing absolute path which can be used during Model constructor call. EXCEPTION: src is
        None or "" in which case nothing will be done and src will be returned
        2. It will add (config_path, model_utils.ArtifactItem()) pair to self.artifacts. If "src" is local existing
        path, then it will be returned in absolute path form. elif "src" starts with "mridc_file:unique_artifact_name":
        .mridc will be untarred to a temporary folder location and an actual existing path will be returned else an
        error will be raised.

        WARNING: use .register_artifact calls in your models' constructors.
        The returned path is not guaranteed to exist after you have exited your model's constructor.

        Parameters
        ----------
        model: ModelPT object to register artifact for.
        config_path: Artifact key. Usually corresponds to the model config.
        src: Path to artifact.
        verify_src_exists: If set to False, then the artifact is optional and register_artifact will return None
         even if src is not found. Defaults to True.

        Returns
        --------
        If src is not None or empty it always returns absolute path which is guaranteed to exist during model instance
         life.
        """
        app_state = AppState()

        artifact_item = mridc.utils.model_utils.ArtifactItem()  # type: ignore

        # This is for backward compatibility, if the src objects exists simply inside the tarfile
        # without its key having been overridden, this pathway will be used.
        src_obj_name = os.path.basename(src)
        if app_state.mridc_file_folder is not None:
            src_obj_path = os.path.abspath(os.path.join(app_state.mridc_file_folder, src_obj_name))
        else:
            src_obj_path = src_obj_name

        # src is a local existing path - register artifact and return exact same path for usage by the model
        if os.path.exists(os.path.abspath(src)):
            return_path = os.path.abspath(src)
            artifact_item.path_type = mridc.utils.model_utils.ArtifactPathType.LOCAL_PATH  # type: ignore

        elif src.startswith("mridc:"):
            return_path = os.path.abspath(os.path.join(app_state.mridc_file_folder, src[5:]))
            artifact_item.path_type = mridc.utils.model_utils.ArtifactPathType.TAR_PATH  # type: ignore

        elif os.path.exists(src_obj_path):
            return_path = src_obj_path
            artifact_item.path_type = mridc.utils.model_utils.ArtifactPathType.TAR_PATH  # type: ignore
        elif verify_src_exists:
            raise FileNotFoundError(
                f"src path does not exist or it is not a path in mridc file. src value I got was: {src}. "
                f"Absolute: {os.path.abspath(src)}"
            )
        else:
            # artifact is optional and we simply return None
            return None

        if not os.path.exists(return_path):
            raise AssertionError

        artifact_item.path = os.path.abspath(src)
        model.artifacts[config_path] = artifact_item
        # we were called by ModelPT
        if hasattr(model, "cfg"):
            with open_dict(model._cfg):
                OmegaConf.update(model.cfg, config_path, return_path)
        return return_path

    def _handle_artifacts(self, model, mridc_file_folder):
        """
        This method is called by ModelPT.save_to() and ModelPT.load_from(). It will handle all artifacts and save them
        to the mridc file.

        Parameters
        ----------
        model: ModelPT object to register artifact for.
        mridc_file_folder: Path to the mridc file.
        """
        tarfile_artifacts = []
        app_state = AppState()
        for conf_path, artiitem in model.artifacts.items():
            if artiitem.path_type == mridc.utils.model_utils.ArtifactPathType.LOCAL_PATH:
                if not os.path.exists(artiitem.path):
                    raise FileNotFoundError(f"Artifact {conf_path} not found at location: {artiitem.path}")

                # Generate new uniq artifact name and copy it to mridc_file_folder
                # Note uuid.uuid4().hex is guaranteed to be 32 character long
                artifact_base_name = os.path.basename(artiitem.path)
                artifact_uniq_name = f"{uuid.uuid4().hex}_{artifact_base_name}"
                shutil.copy2(artiitem.path, os.path.join(mridc_file_folder, artifact_uniq_name))

                # Update artifacts registry
                artiitem.hashed_path = f"mridc:{artifact_uniq_name}"
                model.artifacts[conf_path] = artiitem

            elif artiitem.path_type == mridc.utils.model_utils.ArtifactPathType.TAR_PATH:
                # process all tarfile artifacts in one go, so preserve key-value pair
                tarfile_artifacts.append((conf_path, artiitem))

            else:
                raise ValueError("Directly referencing artifacts from other mridc files isn't supported yet")

        # Process current tarfile artifacts by unpacking the previous tarfile and extract the artifacts
        # that are currently required.
        model_metadata = app_state.get_model_metadata_from_guid(model.model_guid)
        if tarfile_artifacts and model_metadata.restoration_path is not None:
            # Need to step into mridc archive to extract file
            # Get path where the command is executed - the artifacts will be "retrieved" there
            # (original .mridc behavior)
            cwd = os.getcwd()
            try:
                # Step into the mridc archive to try and find the file
                with tempfile.TemporaryDirectory() as archive_dir:
                    self._unpack_mridc_file(path2file=model_metadata.restoration_path, out_folder=archive_dir)
                    os.chdir(archive_dir)
                    for conf_path, artiitem in tarfile_artifacts:
                        # Get basename and copy it to mridc_file_folder
                        if "mridc:" in artiitem.path:
                            artifact_base_name = artiitem.path.split("mridc:")[1]
                        else:
                            artifact_base_name = os.path.basename(artiitem.path)
                        # no need to hash here as we are in tarfile_artifacts which are already hashed
                        artifact_uniq_name = artifact_base_name
                        shutil.copy2(artifact_base_name, os.path.join(mridc_file_folder, artifact_uniq_name))

                        # Update artifacts registry
                        new_artiitem = mridc.utils.model_utils.ArtifactItem()
                        new_artiitem.path = f"mridc:{artifact_uniq_name}"
                        new_artiitem.path_type = mridc.utils.model_utils.ArtifactPathType.TAR_PATH
                        model.artifacts[conf_path] = new_artiitem
            finally:
                # change back working directory
                os.chdir(cwd)

    @staticmethod
    def _update_artifact_paths(model, path2yaml_file):
        """
        This method is called by ModelPT.save_to() and ModelPT.load_from() to update the artifact paths in the
        model.
        """
        if model.artifacts is not None and len(model.artifacts) > 0:
            conf = OmegaConf.load(path2yaml_file)
            for conf_path, item in model.artifacts.items():
                if item.hashed_path is None:
                    OmegaConf.update(conf, conf_path, item.path)
                else:
                    OmegaConf.update(conf, conf_path, item.hashed_path)
            with open(path2yaml_file, "w", encoding="utf-8") as fout:
                OmegaConf.save(config=conf, f=fout, resolve=True)

    @staticmethod
    def _inject_model_parallel_rank_for_ckpt(dirname, basename):
        """
        This method is called by ModelPT.save_to() and ModelPT.load_from() to inject the parallel rank of the process
        into the checkpoint file name.
        """
        model_weights = os.path.join(dirname, basename)
        model_weights = mridc.utils.model_utils.inject_model_parallel_rank(model_weights)
        return model_weights

    @staticmethod
    def _make_mridc_file_from_folder(filename, source_dir):
        """This method is called by ModelPT.save_to() and ModelPT.load_from() to create a mridc file from a folder."""
        dirname = os.path.dirname(filename)
        os.makedirs(dirname, exist_ok=True)
        with tarfile.open(filename, "w") as tar:
            tar.add(source_dir, arcname=".")

    @staticmethod
    def _unpack_mridc_file(path2file: str, out_folder: str) -> str:
        """This method is called by ModelPT.save_to() and ModelPT.load_from() to unpack a mridc file."""
        if not os.path.exists(path2file):
            raise FileNotFoundError(f"{path2file} does not exist")
        # we start with an assumption of uncompressed tar, which should be true for versions 1.7.0 and above
        tar_header = "r:"
        try:
            tar_test = tarfile.open(path2file, tar_header)
            tar_test.close()
        except tarfile.ReadError:
            # can be older checkpoint => try compressed tar
            tar_header = "r:gz"
        tar = tarfile.open(path2file, tar_header)
        tar.extractall(path=out_folder)
        tar.close()
        return out_folder

    @staticmethod
    def _save_state_dict_to_disk(state_dict, filepath):
        """This method is called by ModelPT.save_to() and ModelPT.load_from() to save the state dict to disk."""
        torch.save(state_dict, filepath)

    @staticmethod
    def _load_state_dict_from_disk(model_weights, map_location=None):
        """This method is called by ModelPT.save_to() and ModelPT.load_from() to load the state dict from disk."""
        return torch.load(model_weights, map_location=map_location)

    @property
    def model_config_yaml(self) -> str:
        """This property is used to get the path to the model config yaml file."""
        return self._model_config_yaml

    @model_config_yaml.setter
    def model_config_yaml(self, path: str):
        """This property is used to set the path to the model config yaml file."""
        self._model_config_yaml = path

    @property
    def model_weights_ckpt(self) -> str:
        """This property is used to get the path to the model weights ckpt file."""
        return self._model_weights_ckpt

    @model_weights_ckpt.setter
    def model_weights_ckpt(self, path: str):
        """This property is used to set the path to the model weights ckpt file."""
        self._model_weights_ckpt = path

    @property
    def model_extracted_dir(self) -> Optional[str]:
        return self._model_extracted_dir

    @model_extracted_dir.setter
    def model_extracted_dir(self, path: None):
        self._model_extracted_dir = path
