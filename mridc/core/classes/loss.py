# coding=utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/core/classes/loss.py

import torch

from mridc.core.classes.common import Serialization, Typing

__all__ = ["Loss"]


class Loss(torch.nn.modules.loss._Loss, Typing, Serialization):  # noqa: WPS600
    """Inherit this class to implement custom loss."""
