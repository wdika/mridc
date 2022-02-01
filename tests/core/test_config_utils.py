# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/wdika/NeMo/blob/main/tests/core/test_config_utils.py
from abc import ABC
from dataclasses import dataclass
from typing import Any

import pytest

from mridc.utils import config_utils


@pytest.fixture()
def cls():
    class DummyClass:
        def __init__(self, a, b=5, c: int = 0, d: "ABC" = None):
            pass

    return DummyClass


class TestConfigUtils:
    @pytest.mark.unit
    def test_all_args_exist(self, cls):
        @dataclass
        class DummyDataClass:
            a: int = -1
            b: int = 5
            c: int = 0
            d: Any = None

        result = config_utils.assert_dataclass_signature_match(cls, DummyDataClass)
        signatures_match, cls_subset, dataclass_subset = result

        assert signatures_match
        assert cls_subset is None
        assert dataclass_subset is None

    @pytest.mark.unit
    def test_extra_args_exist_but_is_ignored(self, cls):
        @dataclass
        class DummyDataClass:
            a: int = -1
            b: int = 5
            c: int = 0
            d: Any = None

        result = config_utils.assert_dataclass_signature_match(cls, DummyDataClass, ignore_args=["e"])
        signatures_match, cls_subset, dataclass_subset = result

        assert signatures_match
        assert cls_subset is None
        assert dataclass_subset is None

    @pytest.mark.unit
    def test_args_exist_but_is_remapped(self, cls):
        @dataclass
        class DummyDataClass:
            a: int = -1
            b: int = 5
            c: int = 0
            e: Any = None  # Assume remapped

        result = config_utils.assert_dataclass_signature_match(cls, DummyDataClass, remap_args={"e": "d"})
        signatures_match, cls_subset, dataclass_subset = result

        assert signatures_match
        assert cls_subset is None
        assert dataclass_subset is None
