# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/tree/main/nemo/utils

from dataclasses import dataclass
from threading import Lock
from typing import Dict, Optional

from mridc.utils.metaclasses import Singleton


@dataclass()
class ModelMetadataRegistry:
    """A registry for model metadata."""

    guid: str
    gidx: int
    restoration_path: Optional[str] = None


class AppState(metaclass=Singleton):
    """A singleton class that holds the state of the application."""

    def __init__(self):
        """Initializes the AppState class."""
        # method call lock
        self.model_parallel_rank = None
        self.__lock = Lock()

        # TODO: should we store global config in hydra_runner?
        self._app_cfg = None

        # World info
        self._device_id = None
        self._local_rank = None
        self._global_rank = None
        self._model_parallel_rank = None
        self._tensor_model_parallel_rank = None
        self._pipeline_model_parallel_rank = None
        self._data_parallel_rank = None

        self._world_size = None
        self._model_parallel_size = None
        self._tensor_model_parallel_size = None
        self._tensor_model_parallel_group = None
        self._pipeline_model_parallel_size = None
        self._pipeline_model_parallel_group = None
        self._pipeline_model_parallel_split_rank = None
        self._model_parallel_group = None
        self._data_parallel_size = None
        self._data_parallel_group = None

        self._random_seed = None

        # Logging info
        self._log_dir = None
        self._exp_dir = None
        self._name = None
        self._checkpoint_name = None
        self._version = None
        self._create_checkpoint_callback = None
        self._checkpoint_callback_params = None

        # Save and Restore (.mridc)
        self._tmpdir_name = None
        self._is_model_being_restored = False
        self._mridc_file_folder = None
        self._model_restore_path = None
        self._all_model_restore_paths = []
        self._model_guid_map = {}  # type: Dict[str, ModelMetadataRegistry]

    @property
    def device_id(self):
        """Property returns the device_id."""
        return self._device_id

    @device_id.setter
    def device_id(self, id: int):
        """Property sets the device_id."""
        self._device_id = id

    @property
    def world_size(self):
        """Property returns the total number of GPUs."""
        return self._world_size

    @world_size.setter
    def world_size(self, size: int):
        """Property sets the total number of GPUs."""
        self._world_size = size

    @property
    def model_parallel_size(self):
        """Property returns the number of GPUs in each model parallel group."""
        return self._model_parallel_size

    @model_parallel_size.setter
    def model_parallel_size(self, size: int):
        """Property sets the number of GPUs in each model parallel group."""
        self._model_parallel_size = size

    @property
    def tensor_model_parallel_size(self):
        """Property returns the number of GPUs in each model parallel group."""
        return self._tensor_model_parallel_size

    @tensor_model_parallel_size.setter
    def tensor_model_parallel_size(self, size):
        """Property sets the number of GPUs in each model parallel group."""
        self._tensor_model_parallel_size = size

    @property
    def pipeline_model_parallel_size(self):
        """Property returns the number of GPUs in each model parallel group."""
        return self._pipeline_model_parallel_size

    @pipeline_model_parallel_size.setter
    def pipeline_model_parallel_size(self, size):
        """Property sets the number of GPUs in each model parallel group."""
        self._pipeline_model_parallel_size = size

    @property
    def data_parallel_size(self):
        """Property returns the number of GPUs in each data parallel group."""
        return self._data_parallel_size

    @data_parallel_size.setter
    def data_parallel_size(self, size: int):
        """Property sets the number of GPUs in each data parallel group."""
        self._data_parallel_size = size

    @property
    def local_rank(self):
        """Property returns the local rank."""
        return self._local_rank

    @local_rank.setter
    def local_rank(self, rank: int):
        """Property sets the local rank."""
        self._local_rank = rank

    @property
    def global_rank(self):
        """Property returns the global rank."""
        return self._global_rank

    @global_rank.setter
    def global_rank(self, rank: int):
        """Property sets the global rank."""
        self._global_rank = rank

    @property
    def tensor_model_parallel_rank(self):
        """Property returns the model parallel rank."""
        return self._tensor_model_parallel_rank

    @tensor_model_parallel_rank.setter
    def tensor_model_parallel_rank(self, rank):
        """Property sets the model parallel rank."""
        self._tensor_model_parallel_rank = rank

    @property
    def tensor_model_parallel_group(self):
        """Property returns the model parallel group."""
        return self._tensor_model_parallel_group

    @tensor_model_parallel_group.setter
    def tensor_model_parallel_group(self, group):
        """Property sets the model parallel group."""
        self._tensor_model_parallel_group = group

    @property
    def pipeline_model_parallel_rank(self):
        """Property returns the model parallel rank."""
        return self._pipeline_model_parallel_rank

    @pipeline_model_parallel_rank.setter
    def pipeline_model_parallel_rank(self, rank):
        """Property sets the model parallel rank."""
        self._pipeline_model_parallel_rank = rank

    @property
    def pipeline_model_parallel_split_rank(self):
        """Property returns the model parallel split rank."""
        return self._pipeline_model_parallel_split_rank

    @pipeline_model_parallel_split_rank.setter
    def pipeline_model_parallel_split_rank(self, rank):
        """Property sets the model parallel split rank."""
        self._pipeline_model_parallel_split_rank = rank

    @property
    def pipeline_model_parallel_group(self):
        """Property returns the model parallel group."""
        return self._pipeline_model_parallel_group

    @pipeline_model_parallel_group.setter
    def pipeline_model_parallel_group(self, group):
        """Property sets the model parallel group."""
        self._pipeline_model_parallel_group = group

    @property
    def data_parallel_rank(self):
        """Property returns the data parallel rank."""
        return self._data_parallel_rank

    @data_parallel_rank.setter
    def data_parallel_rank(self, rank: int):
        """Property sets the data parallel rank."""
        self._data_parallel_rank = rank

    @property
    def data_parallel_group(self):
        """Property returns the data parallel group."""
        return self._data_parallel_group

    @data_parallel_group.setter
    def data_parallel_group(self, group):
        """Property sets the data parallel group."""
        self._data_parallel_group = group

    @property
    def random_seed(self):
        """Property returns the random seed."""
        return self._random_seed

    @random_seed.setter
    def random_seed(self, seed: int):
        """Property sets the random seed."""
        self._random_seed = seed

    @property
    def log_dir(self):
        """Returns the log_dir set by exp_manager."""
        return self._log_dir

    @log_dir.setter
    def log_dir(self, dir):
        """Sets the log_dir property."""
        self._log_dir = dir

    @property
    def exp_dir(self):
        """Returns the exp_dir set by exp_manager."""
        return self._exp_dir

    @exp_dir.setter
    def exp_dir(self, dir):
        """Sets the log_dir property."""
        self._exp_dir = dir

    @property
    def name(self):
        """Returns the name set by exp_manager."""
        return self._name

    @name.setter
    def name(self, name: str):
        """Sets the name property."""
        self._name = name

    @property
    def checkpoint_name(self):
        """Returns the name set by exp_manager."""
        return self._checkpoint_name

    @checkpoint_name.setter
    def checkpoint_name(self, name: str):
        """Sets the name property."""
        self._checkpoint_name = name

    @property
    def version(self):
        """Returns the version set by exp_manager."""
        return self._version

    @version.setter
    def version(self, version: str):
        """Sets the version property."""
        self._version = version

    @property
    def create_checkpoint_callback(self):
        """Returns the create_checkpoint_callback set by exp_manager."""
        return self._create_checkpoint_callback

    @create_checkpoint_callback.setter
    def create_checkpoint_callback(self, create_checkpoint_callback: bool):
        """Sets the create_checkpoint_callback property."""
        self._create_checkpoint_callback = create_checkpoint_callback

    @property
    def checkpoint_callback_params(self):
        """Returns the version set by exp_manager."""
        return self._checkpoint_callback_params

    @checkpoint_callback_params.setter
    def checkpoint_callback_params(self, params: dict):
        """Sets the name property."""
        self._checkpoint_callback_params = params

    @property
    def model_restore_path(self):
        """Returns the model_restore_path set by exp_manager."""
        return self._all_model_restore_paths[-1] if len(self._all_model_restore_paths) > 0 else None

    @model_restore_path.setter
    def model_restore_path(self, path):
        """Sets the model_restore_path property."""
        with self.__lock:
            self._model_restore_path = path
            self._all_model_restore_paths.append(path)

    def register_model_guid(self, guid: str, restoration_path: Optional[str] = None):
        """Maps a guid to its restore path (None or last absolute path)."""
        with self.__lock:
            if guid in self._model_guid_map:
                idx = self._model_guid_map[guid].gidx
            else:
                idx = len(self._model_guid_map)
            self._model_guid_map[guid] = ModelMetadataRegistry(guid, idx, restoration_path=restoration_path)

    def reset_model_guid_registry(self):
        """Resets the model guid registry."""
        with self.__lock:
            self._model_guid_map.clear()

    def get_model_metadata_from_guid(self, guid) -> ModelMetadataRegistry:
        """Returns the global model idx and restoration path."""
        return self._model_guid_map[guid]

    @property
    def is_model_being_restored(self) -> bool:
        """Returns whether a model is being restored."""
        return self._is_model_being_restored

    @is_model_being_restored.setter
    def is_model_being_restored(self, is_restored: bool):
        """Sets whether a model is being restored."""
        self._is_model_being_restored = is_restored

    @property
    def mridc_file_folder(self) -> str:
        """Returns the mridc_file_folder set by exp_manager."""
        return self._mridc_file_folder

    @mridc_file_folder.setter
    def mridc_file_folder(self, path: str):
        """Sets the mridc_file_folder property."""
        self._mridc_file_folder = path
