# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from abc import ABC

from mridc.collections.motioncompensation.modules.decolearn import DeCoLearn
from mridc.collections.reconstruction.models.cirim import CIRIM

__all__ = ["DeCoCIRIM"]


class DeCoCIRIM(DeCoLearn, CIRIM, ABC):
    """
    Implementation of the Deformation-Compensated Cascades of Independently Recurrent Inference Machines.
    """
