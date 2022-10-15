# coding=utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/core/neural_types/comparison.py

from enum import Enum

__all__ = ["NeuralTypeComparisonResult"]


class NeuralTypeComparisonResult(Enum):
    """The result of comparing two neural type objects for compatibility. When comparing A.compare_to(B)."""

    SAME = 0
    LESS = 1  # A is B
    GREATER = 2  # B is A
    DIM_INCOMPATIBLE = 3  # Resize connector might fix incompatibility
    # A transpose and/or converting between lists and tensors will make them same
    TRANSPOSE_SAME = 4
    CONTAINER_SIZE_MISMATCH = 5  # A and B contain different number of elements
    INCOMPATIBLE = 6  # A and B are incompatible
    # A and B are of the same type but parametrized differently
    SAME_TYPE_INCOMPATIBLE_PARAMS = 7
    UNCHECKED = 8  # type comparison wasn't done
