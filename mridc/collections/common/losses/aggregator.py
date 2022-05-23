# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/common/losses/aggregator.py

from typing import List

import torch

__all__ = ["AggregatorLoss"]

from mridc.core.classes.common import typecheck
from mridc.core.classes.loss import Loss
from mridc.core.neural_types.elements import LossType
from mridc.core.neural_types.neural_type import NeuralType


class AggregatorLoss(Loss):
    """
    Sums several losses into one.

    Parameters
    ----------
    num_inputs: number of input losses
    weights: a list of coefficient for merging losses
    """

    @property
    def input_types(self):
        """Returns definitions of module input ports."""
        return {f"loss_{str(i + 1)}": NeuralType(elements_type=LossType()) for i in range(self._num_losses)}

    @property
    def output_types(self):
        """Returns definitions of module output ports."""
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self, num_inputs: int = 2, weights: List[float] = None):
        super().__init__()
        self._num_losses = num_inputs
        if weights is not None and len(weights) != num_inputs:
            raise ValueError("Length of weights should be equal to the number of inputs (num_inputs)")

        self._weights = weights

    @typecheck()
    def forward(self, **kwargs):
        """Computes the sum of the losses."""
        values = [kwargs[x] for x in sorted(kwargs.keys())]
        loss = torch.zeros_like(values[0])
        for loss_idx, loss_value in enumerate(values):
            if self._weights is not None:
                loss = loss.add(loss_value, alpha=self._weights[loss_idx])
            else:
                loss = loss.add(loss_value)
        return loss
