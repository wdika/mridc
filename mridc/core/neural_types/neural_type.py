# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/core/neural_types/neural_type.py

from typing import Optional, Tuple

__all__ = ["NeuralType", "NeuralTypeError", "NeuralPortNameMismatchError", "NeuralPortNmTensorMismatchError"]

from mridc.core.neural_types.axes import AxisKind, AxisType
from mridc.core.neural_types.comparison import NeuralTypeComparisonResult
from mridc.core.neural_types.elements import ElementType, VoidType


class NeuralType:
    """
    This is the main class which would represent neural type concept. It is used to represent *the types* of inputs and
     outputs.

    Parameters
    ----------
    axes: a tuple of AxisTypes objects representing the semantics of what varying each axis means. You can use a short,
     string-based form here. For example: ('B', 'C', 'H', 'W') would correspond to an NCHW format frequently used in
     computer vision. ('B', 'T', 'D') is frequently used for signal processing and means
     [batch, time, dimension/channel].
    elements_type: an instance of ElementType class representing the semantics of what is stored inside the tensor.
    For example: logits (LogitsType), log probabilities (LogprobType), etc.
    optional: By default, this is false. If set to True, it would mean that input to the port of this type can be
    optional.
    """

    def __str__(self):
        if self.axes is not None:
            return f"axes: {self.axes}; elements_type: {self.elements_type.__class__.__name__}"
        return f"axes: None; elements_type: {self.elements_type.__class__.__name__}"

    def __init__(self, axes: Optional[Tuple] = None, elements_type: ElementType = VoidType(), optional=False):
        if not isinstance(elements_type, ElementType):
            raise ValueError(
                "elements_type of NeuralType must be an instance of a class derived from ElementType. "
                "Did you pass a class instead?"
            )
        self.elements_type = elements_type
        if axes is not None:
            NeuralType.__check_sanity(axes)
            axes_list = []
            for axis in axes:
                if isinstance(axis, str):
                    axes_list.append(AxisType(AxisKind.from_str(axis), None))
                elif isinstance(axis, AxisType):
                    axes_list.append(axis)
                else:
                    raise ValueError("axis type must be either str or AxisType instance")
            self.axes = tuple(axes_list)  # type: ignore
        else:
            self.axes = None  # type: ignore
        self.optional = optional

    def compare(self, second) -> NeuralTypeComparisonResult:
        """
        Performs neural type comparison of self with second. When you chain two modules' inputs/outputs via __call__
        method, this comparison will be called to ensure neural type compatibility.
        """
        # First, handle dimensionality
        axes_a = self.axes
        axes_b = second.axes

        # "Big void" type
        if isinstance(self.elements_type, VoidType) and self.axes is None:
            return NeuralTypeComparisonResult.SAME

        if self.axes is None:
            if second.axes is None:
                return self.elements_type.compare(second.elements_type)
            return NeuralTypeComparisonResult.INCOMPATIBLE

        dimensions_pass = NeuralType.__compare_axes(axes_a, axes_b)  # type: ignore
        element_comparison_result = self.elements_type.compare(second.elements_type)

        # SAME DIMS
        if dimensions_pass == 0:
            return element_comparison_result
        # TRANSPOSE_SAME DIMS
        if dimensions_pass == 1 and element_comparison_result == NeuralTypeComparisonResult.SAME:
            return NeuralTypeComparisonResult.TRANSPOSE_SAME
        if (
            dimensions_pass == 1
            or dimensions_pass == 2
            and element_comparison_result != NeuralTypeComparisonResult.SAME
        ):
            return NeuralTypeComparisonResult.INCOMPATIBLE
        if dimensions_pass == 2:
            return NeuralTypeComparisonResult.DIM_INCOMPATIBLE
        return NeuralTypeComparisonResult.INCOMPATIBLE

    def compare_and_raise_error(self, parent_type_name, port_name, second_object):
        """Method compares definition of one type with another and raises an error if not compatible."""
        type_compatibility = self.compare(second_object)
        if type_compatibility not in (NeuralTypeComparisonResult.SAME, NeuralTypeComparisonResult.GREATER):
            raise NeuralPortNmTensorMismatchError(
                parent_type_name, port_name, str(self), str(second_object.ntype), type_compatibility
            )

    def __eq__(self, other):
        """Checks if two NeuralTypes are equal."""
        return self.compare(other) if isinstance(other, NeuralType) else False

    @staticmethod
    def __check_sanity(axes):
        """Check that list come before any tensor dimension"""
        are_strings = True
        for axis in axes:
            if not isinstance(axis, str):
                are_strings = False
            if isinstance(axis, str) and not are_strings:
                raise ValueError("Either use full class names or all strings")
        if are_strings:
            return
        checks_passed = True
        saw_tensor_dim = False
        for axis in axes:
            if not axis.is_list:
                saw_tensor_dim = True
            elif saw_tensor_dim:  # which is preceded by tensor dim
                checks_passed = False
        if not checks_passed:
            raise ValueError(
                "You have list dimension after Tensor dimension. All list dimensions must preceded Tensor dimensions"
            )

    @staticmethod
    def __compare_axes(axes_a, axes_b) -> int:
        """
        Compares axes_a and axes_b
        Args:
            axes_a: first axes tuple
            axes_b: second axes tuple
        Returns:
            0 - if they are exactly the same
            1 - if they are "TRANSPOSE_SAME"
            2 - if they are "DIM_INCOMPATIBLE"
            3 - if they are different
        """
        if axes_a is None:
            return 0 if axes_b is None else 3
        if axes_b is None:
            return 3
        if len(axes_a) != len(axes_b):
            return 3
        # After these ifs we know that len(axes_a) == len(axes_b)

        same = True
        kinds_a = {}
        kinds_b = {}
        for axis_a, axis_b in zip(axes_a, axes_b):
            kinds_a[axis_a.kind] = axis_a.size
            kinds_b[axis_b.kind] = axis_b.size
            if axis_a.kind == AxisKind.Any:
                same = True
            elif (
                axis_a.kind != axis_b.kind
                or axis_a.is_list != axis_b.is_list
                or (axis_a.size != axis_b.size and axis_a.size is not None)
            ):
                same = False
        if same:
            return 0
        # can be TRANSPOSE_SAME, DIM_INCOMPATIBLE
        if kinds_a.keys() == kinds_b.keys():
            return next((2 for key, value in kinds_a.items() if kinds_b[key] != value), 1)
        return 3

    def __repr__(self):
        """Returns string representation of NeuralType."""
        axes = str(self.axes) if self.axes is not None else "None"
        if self.elements_type is not None:
            element_type = repr(self.elements_type)
        else:
            element_type = "None"

        data = f"axis={axes}, element_type={element_type}"

        if self.optional:
            data = f"{data}, optional={self.optional}"

        return f"{self.__class__.__name__}({data})"


class NeuralTypeError(Exception):
    """Base class for neural type related exceptions."""


class NeuralPortNameMismatchError(NeuralTypeError):
    """Exception raised when neural module is called with incorrect port names."""

    def __init__(self, input_port_name):
        super().__init__()
        self.message = "Wrong input port name: {0}".format(input_port_name)


class NeuralPortNmTensorMismatchError(NeuralTypeError):
    """Exception raised when a port is fed with a NmTensor of incompatible type."""

    def __init__(self, class_name, port_name, first_type, second_type, type_compatibility):
        super().__init__()
        self.message = "\nIn {}. \nPort: {} and a NmTensor it was fed are \n".format(
            class_name, port_name
        ) + "of incompatible neural types:\n\n{} \n\n and \n\n{}".format(first_type, second_type)

        self.message += "\n\nType comparison result: {}".format(type_compatibility)
