# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from mridc.collections.multitask.rs.parts.transforms import RSMRIDataTransforms

__all__ = ["SegmentationMRIDataTransforms"]


class SegmentationMRIDataTransforms(RSMRIDataTransforms):
    """Transforms for the segmentation task."""
