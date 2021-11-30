# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/core/neural_types/axes.py

from enum import Enum
from typing import Optional

__all__ = ["AxisKindAbstract", "AxisKind", "AxisType"]


class AxisKindAbstract(Enum):
    """
    This is an abstract Enum to represents what does varying axis dimension mean. In practice, you will almost always
    use AxisKind Enum. This Enum should be inherited by your OWN Enum if you aren't satisfied with AxisKind. Then your
    own Enum can be used instead of AxisKind.
    """


class AxisKind(AxisKindAbstract):
    """
    This Enum represents what does varying axis dimension mean. For example, does this dimension correspond to width, \
     batch, time, etc. The "Dimension" and "Channel" kinds are the same and used to represent a general axis. "Any" \
     axis will accept any axis kind fed to it.
    """

    # TODO (wdika): change names of the enums
    Batch = 0
    Time = 1
    Dimension = 2
    Channel = 2
    Width = 3
    Height = 4
    Any = 5
    Sequence = 6
    FlowGroup = 7
    Singleton = 8  # Used to represent a axis that has size 1

    def __repr__(self):
        """Returns short string representation of the AxisKind"""
        return self.__str__()

    def __str__(self):
        """Returns short string representation of the AxisKind"""
        return str(self.name).lower()

    def t_with_string(self, text):
        """It checks if text is 't_<any string>'"""
        return text.startswith("t_") and text.endswith("_") and text[2:-1] == self.__str__()

    @staticmethod
    def from_str(label):
        """Returns AxisKind instance based on short string representation"""
        _label = label.lower().strip()
        if _label in ("b", "n", "batch"):
            return AxisKind.Batch
        if _label == "t" or _label == "time" or (len(_label) > 2 and _label.startswith("t_")):
            return AxisKind.Time
        if _label in ("d", "c", "channel"):
            return AxisKind.Dimension
        if _label in ("w", "width"):
            return AxisKind.Width
        if _label in ("h", "height"):
            return AxisKind.Height
        if _label in ("s", "singleton"):
            return AxisKind.Singleton
        if _label in ("seq", "sequence"):
            return AxisKind.Sequence
        if _label == "flowgroup":
            return AxisKind.FlowGroup
        if _label == "any":
            return AxisKind.Any
        raise ValueError(f"Can't create AxisKind from {label}")


class AxisType:
    """This class represents axis semantics and (optionally) it's dimensionality

    Parameters
    ----------
    kind: what kind of axis it is? For example Batch, Height, etc.
        AxisKindAbstract
    size: specify if the axis should have a fixed size. By default, it is set to None and you typically do not want to
    set it for Batch and Time.
        (int, optional)
    is_list: whether this is a list or a tensor axis.
        (bool, default=False)
    """

    def __init__(self, kind: AxisKindAbstract, size: Optional[int] = None, is_list=False):
        if size is not None and is_list:
            raise ValueError("The axis can't be list and have a fixed size")
        self.kind = kind
        self.size = size
        self.is_list = is_list

    def __repr__(self):
        """Returns short string representation of the AxisType"""
        if self.size is None:
            representation = str(self.kind)
        else:
            representation = f"{str(self.kind)}:{self.size}"
        if self.is_list:
            representation += "_listdim"
        return representation
