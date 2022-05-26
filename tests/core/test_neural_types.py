# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/tests/core/test_neural_types.py

import pytest

from mridc.core.neural_types.axes import AxisKind, AxisType
from mridc.core.neural_types.comparison import NeuralTypeComparisonResult
from mridc.core.neural_types.elements import ElementType, VoidType
from mridc.core.neural_types.neural_type import NeuralType


class TestNeuralTypeSystem:
    @pytest.mark.unit
    def test_transpose_same_1(self):
        type1 = NeuralType(axes=("B", "T", "C"))
        type2 = NeuralType(axes=("T", "B", "C"))

        assert type1.compare(type2) == NeuralTypeComparisonResult.TRANSPOSE_SAME
        assert type2.compare(type1) == NeuralTypeComparisonResult.TRANSPOSE_SAME

    @pytest.mark.unit
    def test_singletone(self):
        loss_output1 = NeuralType(axes=None)
        loss_output2 = NeuralType(axes=None)

        assert loss_output1.compare(loss_output2) == NeuralTypeComparisonResult.SAME
        assert loss_output2.compare(loss_output1) == NeuralTypeComparisonResult.SAME

    @pytest.mark.unit
    def test_struct(self):
        class BoundingBox(ElementType):
            def __str__(self):
                return "bounding box from detection model"

            def fields(self):
                return ("X", "Y", "W", "H")

        T1 = NeuralType(
            elements_type=BoundingBox(),
            axes=(AxisType(kind=AxisKind.Batch, size=None, is_list=True),),
        )

        class BadBoundingBox(ElementType):
            def __str__(self):
                return "bad bounding box from detection model"

            def fields(self):
                return ("X", "Y", "H")

        T2 = NeuralType(
            elements_type=BadBoundingBox(),
            axes=(AxisType(kind=AxisKind.Batch, size=None, is_list=True),),
        )
        assert T2.compare(T1) == NeuralTypeComparisonResult.INCOMPATIBLE
