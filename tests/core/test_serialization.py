# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/tests/core/test_serialization.py

import pytest
from omegaconf import DictConfig

from mridc.core.classes.common import Serialization


def get_class_path(cls):
    return f"{cls.__module__}.{cls.__name__}"


class MockSerializationImpl(Serialization):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.value = self.__class__.__name__


class MockSerializationImplV2(MockSerializationImpl):
    pass


class TestSerialization:
    @pytest.mark.unit
    def test_self_class_instantiation(self):
        # Target class is V1 impl, calling class is V1 (same class)
        config = DictConfig({"target": get_class_path(MockSerializationImpl)})
        obj = MockSerializationImpl.from_config_dict(config=config)  # Serialization is base class
        new_config = obj.to_config_dict()
        assert config == new_config
        assert isinstance(obj, MockSerializationImpl)
        assert obj.value == "MockSerializationImpl"

    @pytest.mark.unit
    def test_sub_class_instantiation(self):
        # Target class is V1 impl, calling class is V2 (sub class)
        config = DictConfig({"target": get_class_path(MockSerializationImpl)})
        obj = MockSerializationImplV2.from_config_dict(config=config)  # Serialization is base class
        new_config = obj.to_config_dict()
        assert config == new_config
        assert isinstance(obj, MockSerializationImplV2)
        assert obj.value == "MockSerializationImplV2"
