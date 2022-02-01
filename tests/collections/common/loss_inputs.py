# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/wdika/NeMo/edit/main/tests/collections/common/loss_inputs.py

from dataclasses import dataclass

import torch

from tests.collections.common.pl_utils import NUM_BATCHES


@dataclass(frozen=True)
class LossInput:
    """
    The input for ``mridc.collections.common.metrics.GlobalAverageLossMetric`` metric tests.

    Args:
        loss_sum_or_avg: a one dimensional float tensor which contains losses for averaging. Each element is either a
            sum or mean of several losses depending on the parameter ``take_avg_loss`` of the
            ``nemo.collections.common.metrics.GlobalAverageLossMetric`` class.
        num_measurements: a one dimensional integer tensor which contains number of measurements which sums or average
            values are in ``loss_sum_or_avg``.
    """

    loss_sum_or_avg: torch.Tensor
    num_measurements: torch.Tensor


NO_ZERO_NUM_MEASUREMENTS = LossInput(
    loss_sum_or_avg=torch.rand(NUM_BATCHES) * 2.0 - 1.0,
    num_measurements=torch.randint(1, 100, (NUM_BATCHES,)),
)

SOME_NUM_MEASUREMENTS_ARE_ZERO = LossInput(
    loss_sum_or_avg=torch.rand(NUM_BATCHES) * 2.0 - 1.0,
    num_measurements=torch.cat(
        (
            torch.randint(1, 100, (NUM_BATCHES // 2,), dtype=torch.int32),
            torch.zeros(NUM_BATCHES - NUM_BATCHES // 2, dtype=torch.int32),
        )
    ),
)

ALL_NUM_MEASUREMENTS_ARE_ZERO = LossInput(
    loss_sum_or_avg=torch.rand(NUM_BATCHES) * 2.0 - 1.0,
    num_measurements=torch.zeros(NUM_BATCHES, dtype=torch.int32),
)
