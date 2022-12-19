# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from mridc.collections.multitask.models.base import BaseMTLMRIModel
from mridc.collections.segmentation.models import JRSCIRIM


class MTLMRIRS(BaseMTLMRIModel, JRSCIRIM):
    pass
