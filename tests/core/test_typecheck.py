# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/wdika/NeMo/blob/main/tests/core/test_typecheck.py

from mridc.core.neural_types.comparison import NeuralTypeComparisonResult


def recursive_assert_shape(x, shape):
    """Perform recursive shape assert"""
    if isinstance(x, (list, tuple)):
        for xi in x:
            recursive_assert_shape(xi, shape)
        return

    if x.shape != shape:
        raise AssertionError


def recursive_assert_homogeneous_type(x, type_val):
    """Perform recursive type homogeneous assert"""
    if isinstance(x, (list, tuple)):
        for xi in x:
            recursive_assert_homogeneous_type(xi, type_val)
        return

    if x.neural_type.compare(type_val) != NeuralTypeComparisonResult.SAME:
        raise AssertionError
