# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/tests/core/test_save_restore.py

import filecmp
import os
import shutil
import tempfile
from typing import Dict, Optional, Set, Union

import pytest
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from mridc.core.classes.modelPT import ModelPT
from mridc.core.connectors import save_restore_connector
from mridc.utils.app_state import AppState


def classpath(cls):
    return f"{cls.__module__}.{cls.__name__}"


def get_dir_size(path="."):
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total


def get_size(path="."):
    if os.path.isfile(path):
        return os.path.getsize(path)
    elif os.path.isdir(path):
        return get_dir_size(path)


def getattr2(object, attr):
    if "." not in attr:
        return getattr(object, attr)
    arr = attr.split(".")
    return getattr2(getattr(object, arr[0]), ".".join(arr[1:]))


class MockModel(ModelPT):
    def __init__(self, cfg, trainer=None):
        super(MockModel, self).__init__(cfg=cfg, trainer=trainer)
        self.w = torch.nn.Linear(10, 1)
        # mock temp file
        if "temp_file" in self.cfg and self.cfg.temp_file is not None:
            self.temp_file = self.register_artifact("temp_file", self.cfg.temp_file)
            with open(self.temp_file, "r", encoding="utf-8") as f:
                self.temp_data = f.readlines()
        else:
            self.temp_file = None
            self.temp_data = None

    def forward(self, x):
        y = self.w(x)
        return y, self.cfg.temp_file

    def setup_training_data(self, train_data_config: Union[DictConfig, Dict]):
        self._train_dl = None

    def setup_validation_data(self, val_data_config: Union[DictConfig, Dict]):
        self._validation_dl = None

    def setup_test_data(self, test_data_config: Union[DictConfig, Dict]):
        self._test_dl = None

    def list_available_models(cls):
        return []


def _mock_model_config():
    conf = {"temp_file": None, "target": classpath(MockModel)}
    conf = OmegaConf.create({"model": conf})
    OmegaConf.set_struct(conf, True)
    return conf


class TestSaveRestore:
    def __test_restore_elsewhere(
        self,
        model: ModelPT,
        attr_for_eq_check: Set[str] = None,
        override_config_path: Optional[Union[str, DictConfig]] = None,
        map_location: Optional[torch.device] = None,
        strict: bool = False,
        return_config: bool = False,
    ):
        """Test's logic:
        1. Save model into temporary folder (save_folder)
        2. Copy .mridc file from save_folder to restore_folder
        3. Delete save_folder
        4. Attempt to restore from .mridc file in restore_folder and compare to original instance
        """
        # Create a new temporary directory
        with tempfile.TemporaryDirectory() as restore_folder:
            with tempfile.TemporaryDirectory() as save_folder:
                save_folder_path = save_folder
                # Where model will be saved
                model_save_path = os.path.join(save_folder, f"{model.__class__.__name__}.mridc")
                model.save_to(save_path=model_save_path)
                # Where model will be restored from
                model_restore_path = os.path.join(restore_folder, f"{model.__class__.__name__}.mridc")
                shutil.copy(model_save_path, model_restore_path)
            # at this point save_folder should not exist
            assert save_folder_path is not None and not os.path.exists(save_folder_path)
            assert not os.path.exists(model_save_path)
            assert os.path.exists(model_restore_path)
            # attempt to restore
            model_copy = model.__class__.restore_from(
                restore_path=model_restore_path,
                map_location=map_location,
                strict=strict,
                return_config=return_config,
                override_config_path=override_config_path,
            )

            if return_config:
                return model_copy

            assert model.num_weights == model_copy.num_weights
            if attr_for_eq_check is not None and attr_for_eq_check:
                for attr in attr_for_eq_check:
                    assert getattr2(model, attr) == getattr2(model_copy, attr)

            return model_copy

    @pytest.mark.unit
    def test_mock_restore_from_config_override_with_OmegaConf(self):
        with tempfile.NamedTemporaryFile("w") as empty_file:
            # Write some data
            empty_file.writelines(["*****\n"])
            empty_file.flush()

            # Update config
            cfg = _mock_model_config()
            cfg.model.temp_file = empty_file.name

            # Create model
            model = MockModel(cfg=cfg.model, trainer=None)
            model = model.to("cpu")

            assert model.temp_file == empty_file.name

            # Inject arbitrary config arguments (after creating model)
            with open_dict(cfg.model):
                cfg.model.xyz = "abc"

            # Save test (with overriden config as OmegaConf object)
            model_copy = self.__test_restore_elsewhere(model, map_location="cpu", override_config_path=cfg)

        # Restore test
        diff = model.w.weight - model_copy.w.weight
        assert diff.mean() <= 1e-9
        assert model_copy.temp_data == ["*****\n"]

        # Test that new config has arbitrary content
        assert model_copy.cfg.xyz == "abc"

    @pytest.mark.unit
    def test_mock_restore_from_config_override_with_yaml(self):
        with tempfile.NamedTemporaryFile("w") as empty_file, tempfile.NamedTemporaryFile("w") as config_file:
            # Write some data
            empty_file.writelines(["*****\n"])
            empty_file.flush()

            # Update config
            cfg = _mock_model_config()
            cfg.model.temp_file = empty_file.name

            # Create model
            model = MockModel(cfg=cfg.model, trainer=None)
            model = model.to("cpu")

            assert model.temp_file == empty_file.name

            # Inject arbitrary config arguments (after creating model)
            with open_dict(cfg.model):
                cfg.model.xyz = "abc"

            # Write new config into file
            OmegaConf.save(cfg, config_file)

            # Save test (with overriden config as OmegaConf object)
            model_copy = self.__test_restore_elsewhere(
                model, map_location="cpu", override_config_path=config_file.name
            )

            # Restore test
            diff = model.w.weight - model_copy.w.weight
            assert diff.mean() <= 1e-9
            assert filecmp.cmp(model.temp_file, model_copy.temp_file)
            assert model_copy.temp_data == ["*****\n"]

            # Test that new config has arbitrary content
            assert model_copy.cfg.xyz == "abc"

    @pytest.mark.unit
    def test_mock_save_to_multiple_times(self):
        with tempfile.NamedTemporaryFile("w") as empty_file, tempfile.TemporaryDirectory() as tmpdir:
            # Write some data
            empty_file.writelines(["*****\n"])
            empty_file.flush()

            # Update config
            cfg = _mock_model_config()
            cfg.model.temp_file = empty_file.name

            # Create model
            model = MockModel(cfg=cfg.model, trainer=None)  # type: MockModel
            model = model.to("cpu")

            assert model.temp_file == empty_file.name

            # Save test
            model.save_to(os.path.join(tmpdir, "save_0.mridc"))
            model.save_to(os.path.join(tmpdir, "save_1.mridc"))
            model.save_to(os.path.join(tmpdir, "save_2.mridc"))

    @pytest.mark.unit
    def test_multiple_model_save_restore_connector(self):
        class MySaveRestoreConnector(save_restore_connector.SaveRestoreConnector):
            def save_to(self, model, save_path: str):
                save_path = save_path.replace(".mridc", "_XYZ.mridc")
                super(MySaveRestoreConnector, self).save_to(model, save_path)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Update config
            cfg = _mock_model_config()
            # Create model
            model = MockModel(cfg=cfg.model, trainer=None)
            model_with_custom_connector = MockModel(cfg=cfg.model, trainer=None)
            model_with_custom_connector._save_restore_connector = MySaveRestoreConnector()
            model_with_custom_connector.save_to(os.path.join(tmpdir, "save_custom.mridc"))

            assert os.path.exists(os.path.join(tmpdir, "save_custom_XYZ.mridc"))
            assert isinstance(model._save_restore_connector, save_restore_connector.SaveRestoreConnector)
            assert isinstance(model_with_custom_connector._save_restore_connector, MySaveRestoreConnector)

            assert type(MockModel._save_restore_connector) == save_restore_connector.SaveRestoreConnector

    @pytest.mark.unit
    def test_restore_from_save_restore_connector(self):
        class MySaveRestoreConnector(save_restore_connector.SaveRestoreConnector):
            def save_to(self, model, save_path: str):
                save_path = save_path.replace(".mridc", "_XYZ.mridc")
                super().save_to(model, save_path)

        class MockModelV2(MockModel):
            pass

        with tempfile.TemporaryDirectory() as tmpdir:
            # Update config
            cfg = _mock_model_config()

            # Create model
            save_path = os.path.join(tmpdir, "save_custom.mridc")
            model_with_custom_connector = MockModel(cfg=cfg.model, trainer=None)
            model_with_custom_connector._save_restore_connector = MySaveRestoreConnector()
            model_with_custom_connector.save_to(save_path)

            assert os.path.exists(os.path.join(tmpdir, "save_custom_XYZ.mridc"))

            restored_model = MockModelV2.restore_from(
                save_path.replace(".mridc", "_XYZ.mridc"), save_restore_connector=MySaveRestoreConnector()
            )
            assert type(restored_model) == MockModelV2
            assert type(restored_model._save_restore_connector) == MySaveRestoreConnector

    @pytest.mark.unit
    def test_mock_model_model_collision(self):
        # The usual pipeline is working just fine.
        cfg = _mock_model_config()
        model = MockModel(cfg=cfg.model, trainer=None)  # type: MockModel
        model = model.to("cpu")

        # Let's create a custom config with a 'model.model' node.
        cfg = _mock_model_config()
        OmegaConf.set_struct(cfg, False)
        cfg.model.model = "aaa"
        OmegaConf.set_struct(cfg, True)

        # Failing due to collision.
        with pytest.raises(ValueError, match="Creating model config node is forbidden"):
            model = MockModel(cfg=cfg.model, trainer=None)  # type: MockModel
            model = model.to("cpu")

    @pytest.mark.unit
    def test_restore_from_save_restore_connector_extracted_dir(self):
        class MySaveRestoreConnector(save_restore_connector.SaveRestoreConnector):
            def save_to(self, model, save_path: str):
                save_path = save_path.replace(".mridc", "_XYZ.mridc")
                super().save_to(model, save_path)

        class MockModelV2(MockModel):
            pass

        with tempfile.TemporaryDirectory() as extracted_tempdir:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Update config
                cfg = _mock_model_config()

                # Create model
                save_path = os.path.join(tmpdir, "save_custom.mridc")
                model_with_custom_connector = MockModel(cfg=cfg.model, trainer=None)
                model_with_custom_connector._save_restore_connector = MySaveRestoreConnector()
                model_with_custom_connector.save_to(save_path)

                mridc_filepath = os.path.join(tmpdir, "save_custom_XYZ.mridc")
                assert os.path.exists(mridc_filepath)

                # extract the contents to this dir apriori
                # simulate by extracting now before calling restore_from
                connector = MySaveRestoreConnector()
                MySaveRestoreConnector._unpack_mridc_file(mridc_filepath, extracted_tempdir)
                assert get_size(extracted_tempdir) > 0

            # delete the old directory and preserve only the new extracted directory (escape scope of old dir)

            # next, set the model's extracted directory path
            connector.model_extracted_dir = extracted_tempdir

            # note, we pass in the "old" mridc_filepath, stored somewhere other than the extracted directory
            # this mridc_filepath is no longer valid, and has been deleted.
            restored_model = MockModelV2.restore_from(mridc_filepath, save_restore_connector=connector)
        assert type(restored_model) == MockModelV2
        assert type(restored_model._save_restore_connector) == MySaveRestoreConnector

        # assert models have correct restoration information and paths
        appstate = AppState()
        original_metadata = appstate.get_model_metadata_from_guid(model_with_custom_connector.model_guid)
        assert original_metadata.restoration_path is None

        restored_metadata = appstate.get_model_metadata_from_guid(restored_model.model_guid)
        assert restored_metadata.restoration_path is not None

        # assert that the restore path was the path of the pre-extracted directory
        # irrespective of whether an old `mridc_filepath` (which doesnt exist anymore) was passed to restore_from.
        assert extracted_tempdir in restored_metadata.restoration_path
        assert extracted_tempdir not in mridc_filepath
        assert not os.path.exists(mridc_filepath)

        # test for parameter equality
        model_with_custom_connector = model_with_custom_connector.to("cpu")
        restored_model = restored_model.to("cpu")

        original_state_dict = model_with_custom_connector.state_dict()
        restored_state_dict = restored_model.state_dict()
        for orig, restored in zip(original_state_dict.keys(), restored_state_dict.keys()):
            assert (original_state_dict[orig] - restored_state_dict[restored]).abs().mean() < 1e-6
