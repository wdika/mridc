# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/core/classes/loss.py

import torch

__all__ = ["Loss"]

from mridc.core.classes.common import Serialization, Typing


class Loss(torch.nn.modules.loss._Loss, Typing, Serialization):
    """Inherit this class to implement custom loss."""
