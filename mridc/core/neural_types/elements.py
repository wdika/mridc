# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/core/neural_types/elements.py

from abc import ABC, ABCMeta
from typing import Dict, Optional, Tuple

__all__ = [
    "ElementType",
    "VoidType",
    "ChannelType",
    "MRISignal",
    "RecurrentsType",
    "LabelsType",
    "LogprobsType",
    "ProbsType",
    "LossType",
    "RegressionValuesType",
    "CategoricalValuesType",
    "PredictionsType",
    "LengthsType",
    "MaskType",
    "Target",
    "ReconstructionTarget",
    "ImageFeatureValue",
    "Index",
    "ImageValue",
    "NormalizedImageValue",
    "StringLabel",
    "StringType",
    "Length",
    "IntType",
    "FloatType",
    "NormalDistributionSamplesType",
    "NormalDistributionMeanType",
    "NormalDistributionLogVarianceType",
    "LogDeterminantType",
    "SequenceToSequenceAlignmentType",
]

from mridc.core.neural_types.comparison import NeuralTypeComparisonResult


class ElementType(ABC):
    """Abstract class defining semantics of the tensor elements. We are relying on Python for inheritance checking"""

    def __str__(self):
        """Override this method to provide a human readable representation of the type"""
        return self.__doc__

    def __repr__(self):
        """Override this method to provide a human readable representation of the type"""
        return self.__class__.__name__

    @property
    def type_parameters(self) -> Dict:
        """
        Override this property to parametrize your type. For example, you can specify 'storage' type such as float,
        int, bool with 'dtype' keyword. Another example, is if you want to represent a signal with a particular
        property (say, sample frequency), then you can put sample_freq->value in there. When two types are compared
        their type_parameters must match."
        """
        return {}

    @property
    def fields(self) -> Optional[Tuple]:
        """
        This should be used to logically represent tuples/structures. For example, if you want to represent a \
        bounding box (x, y, width, height) you can put a tuple with names ('x', y', 'w', 'h') in here. Under the \
        hood this should be converted to the last tensor dimension of fixed size = len(fields). When two types are \
        compared their fields must match.
        """
        return None

    def compare(self, second) -> NeuralTypeComparisonResult:
        """Override this method to provide a comparison between two types."""
        # First, check general compatibility
        first_t = type(self)
        second_t = type(second)

        if first_t == second_t:
            result = NeuralTypeComparisonResult.SAME
        elif issubclass(first_t, second_t):
            result = NeuralTypeComparisonResult.LESS
        elif issubclass(second_t, first_t):
            result = NeuralTypeComparisonResult.GREATER
        else:
            result = NeuralTypeComparisonResult.INCOMPATIBLE

        if result != NeuralTypeComparisonResult.SAME:
            return result
        # now check that all parameters match
        check_params = set(self.type_parameters.keys()) == set(second.type_parameters.keys())
        if not check_params:
            return NeuralTypeComparisonResult.SAME_TYPE_INCOMPATIBLE_PARAMS
        for k1, v1 in self.type_parameters.items():
            if v1 is None or second.type_parameters[k1] is None:
                # Treat None as Void
                continue
            if v1 != second.type_parameters[k1]:
                return NeuralTypeComparisonResult.SAME_TYPE_INCOMPATIBLE_PARAMS
                # check that all fields match
        if self.fields == second.fields:
            return NeuralTypeComparisonResult.SAME
        return NeuralTypeComparisonResult.INCOMPATIBLE


class VoidType(ElementType):
    """
    Void-like type which is compatible with everything. It is a good practice to use this type only as necessary.
    For example, when you need template-like functionality.
    """

    def compare(cls, second: ABCMeta) -> NeuralTypeComparisonResult:
        """Void type is compatible with everything."""
        return NeuralTypeComparisonResult.SAME


# TODO: Consider moving these files elsewhere
class ChannelType(ElementType):
    """Element to represent convolutional input/output channel."""


class RecurrentsType(ElementType):
    """Element type to represent recurrent layers"""


class LengthsType(ElementType):
    """Element type representing lengths of something"""


class ProbsType(ElementType):
    """Element type to represent probabilities. For example, outputs of softmax layers."""


class LogprobsType(ElementType):
    """Element type to represent log-probabilities. For example, outputs of log softmax layers."""


class LossType(ElementType):
    """Element type to represent outputs of Loss modules"""


class MRISignal(ElementType):
    """
    Element type to represent encoded representation returned by the mri model

    Parameters
    ----------
    freq: sampling frequency of a signal. Note that two signals will only be the same if their freq is the same.
    """

    def __init__(self, freq: int = None):
        self._params = {"freq": freq}

    @property
    def type_parameters(self):
        """Returns the type parameters of the element type."""
        return self._params


class LabelsType(ElementType):
    """Element type to represent labels of something. For example, labels of a dataset."""


class PredictionsType(LabelsType):
    """Element type to represent some sort of predictions returned by model"""


class RegressionValuesType(PredictionsType):
    """Element type to represent labels for regression task"""


class CategoricalValuesType(PredictionsType):
    """Element type to represent labels for categorical classification task"""


class MaskType(PredictionsType):
    """Element type to represent a boolean mask"""


class Index(ElementType):
    """Type representing an element being an index of the sample."""


class Target(ElementType):
    """Type representing an element being a target value."""


class ReconstructionTarget(Target):
    """
    Type representing an element being target value in the reconstruction task, i.e. identifier of a desired
    class.
    """


class ImageValue(ElementType):
    """Type representing an element/value of a single image channel,"""


class NormalizedImageValue(ImageValue):
    """Type representing an element/value of a single image channel normalized to <0-1> range."""


class ImageFeatureValue(ImageValue):
    """Type representing an element (single value) of a (image) feature maps."""


class StringType(ElementType):
    """Element type representing a single string"""


class StringLabel(StringType):
    """Type representing a label being a string with class name (e.g. the "hamster" class in CIFAR100)."""


class BoolType(ElementType):
    """Element type representing a single integer"""


class IntType(ElementType):
    """Element type representing a single integer"""


class FloatType(ElementType):
    """Element type representing a single float"""


class Length(IntType):
    """Type representing an element storing a "length" (e.g. length of a list)."""


class ProbabilityDistributionSamplesType(ElementType):
    """Element to represent tensors that meant to be sampled from a valid probability distribution"""


class NormalDistributionSamplesType(ProbabilityDistributionSamplesType):
    """Element to represent tensors that meant to be sampled from a valid normal distribution"""


class SequenceToSequenceAlignmentType(ElementType):
    """
    Class to represent the alignment from seq-to-seq attention outputs. Generally a mapping from encoder time steps
    to decoder time steps.
    """


class NormalDistributionMeanType(ElementType):
    """Element to represent the mean of a normal distribution"""


class NormalDistributionLogVarianceType(ElementType):
    """Element to represent the log variance of a normal distribution"""


class LogDeterminantType(ElementType):
    """Element for representing log determinants usually used in flow models"""
