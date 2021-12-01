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
    """Create a class with a config attribute."""

    class DummyClass:
        """Dummy class."""

        def __init__(self, a, b=5, c: int = 0, d: "ABC" = None):
            pass

    return DummyClass


class TestConfigUtils:
    """Test the config utils."""

    @pytest.mark.unit
    def test_all_args_exist(self, cls):
        """Test that all arguments exist in the dataclass."""

        @dataclass
        class DummyDataClass:
            """Dummy data class."""

            a: int = -1
            b: int = 5
            c: int = 0
            d: Any = None

        result = config_utils.assert_dataclass_signature_match(cls, DummyDataClass)
        signatures_match, cls_subset, dataclass_subset = result

        if not signatures_match:
            raise AssertionError
        if cls_subset is not None:
            raise AssertionError
        if dataclass_subset is not None:
            raise AssertionError

    @pytest.mark.unit
    def test_extra_args_exist_but_is_ignored(self, cls):
        """Test that extra arguments exist in the dataclass."""

        @dataclass
        class DummyDataClass:
            """Dummy data class."""

            a: int = -1
            b: int = 5
            c: int = 0
            d: Any = None

        result = config_utils.assert_dataclass_signature_match(cls, DummyDataClass, ignore_args=["e"])
        signatures_match, cls_subset, dataclass_subset = result

        if not signatures_match:
            raise AssertionError
        if cls_subset is not None:
            raise AssertionError
        if dataclass_subset is not None:
            raise AssertionError

    @pytest.mark.unit
    def test_args_exist_but_is_remapped(self, cls):
        """Test that arguments exist in the dataclass but are remapped."""

        @dataclass
        class DummyDataClass:
            """Dummy data class."""

            a: int = -1
            b: int = 5
            c: int = 0
            e: Any = None  # Assume remapped

        result = config_utils.assert_dataclass_signature_match(cls, DummyDataClass, remap_args={"e": "d"})
        signatures_match, cls_subset, dataclass_subset = result

        if not signatures_match:
            raise AssertionError
        if cls_subset is not None:
            raise AssertionError
        if dataclass_subset is not None:
            raise AssertionError
