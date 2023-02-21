# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from mridc.collections.common.parts.transforms import MRIDataTransforms

__all__ = ["ReconstructionMRIDataTransforms"]


class ReconstructionMRIDataTransforms(MRIDataTransforms):
    """Transforms for the accelerated-MRI reconstruction task."""
