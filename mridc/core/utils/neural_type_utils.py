# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/core/utils/neural_type_utils.py

from collections import defaultdict

from mridc.core.neural_types.axes import AxisKind
from mridc.core.neural_types.neural_type import NeuralType


def get_io_names(types, disabled_names):
    """
    This method will return a list of input and output names for a given NeuralType.

    Parameters
    ----------
    types: The NeuralType of the module or model to be inspected.
    disabled_names: A list of names that should be excluded from the result.

    Returns
    -------
    A list of input and output names.
    """
    names = list(types.keys())
    for name in disabled_names:
        if name in names:
            names.remove(name)
    return names


def extract_dynamic_axes(name: str, ntype: NeuralType):
    """
    This method will extract BATCH and TIME dimension ids from each provided input/output name argument.

    For example, if module/model accepts argument named "input_signal" with type corresponding to [Batch, Time, Dim]
    shape, then the returned result should contain "input_signal" -> [0, 1] because Batch and Time are dynamic axes
    as they can change from call to call during inference.

    Parameters
    ----------
    name: Name of input or output parameter
    ntype: Corresponding Neural Type

    Returns
    -------
    A dictionary with input/output name as key and a list of dynamic axes as value.
    """

    def unpack_nested_neural_type(neural_type):
        """
        This method will unpack nested NeuralTypes.

        Parameters
        ----------
        neural_type: The NeuralType to be unpacked.

        Returns
        -------
        A list of all the nested NeuralTypes.
        """
        if type(neural_type) in (list, tuple):
            return unpack_nested_neural_type(neural_type[0])
        return neural_type

    dynamic_axes = defaultdict(list)
    if type(ntype) in (list, tuple):
        ntype = unpack_nested_neural_type(ntype)

    if ntype.axes:
        for ind, axis in enumerate(ntype.axes):
            if axis.kind in [AxisKind.Batch, AxisKind.Time, AxisKind.Width, AxisKind.Height]:
                dynamic_axes[name].append(ind)
    return dynamic_axes


def get_dynamic_axes(types, names):
    """
    This method will return a dictionary with input/output names as keys and a list of dynamic axes as values.

    Parameters
    ----------
    types: The NeuralType of the module or model to be inspected.
    names: A list of names that should be inspected.

    Returns
    -------
    A dictionary with input/output names as keys and a list of dynamic axes as values.
    """
    dynamic_axes = defaultdict(list)
    for name in names:
        if name in types:
            dynamic_axes |= extract_dynamic_axes(name, types[name])
    return dynamic_axes
