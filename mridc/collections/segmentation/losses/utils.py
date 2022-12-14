# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from typing import Any, Tuple

import torch
from torch import Tensor

# Taken and adapted from: https://github.com/Project-MONAI/MONAI/blob/dev/monai/metrics/utils.py


def do_metric_reduction(f: torch.Tensor, reduction: str = "mean") -> Tuple[Tensor, Any]:
    """
    Utility function to perform metric reduction.

    Parameters
    ----------
    f: torch.Tensor
        the metric to reduce.
    reduction: str
        the reduction method, default is ``mean``.

    Returns
    -------
    torch.Tensor or Any
        the reduced metric.
    Any
        NaNs if there are any NaNs in the input, otherwise 0.
    """

    # some elements might be Nan (if ground truth y was missing (zeros))
    # we need to account for it
    nans = torch.isnan(f)
    not_nans = (~nans).float()

    t_zero = torch.zeros(1, device=f.device, dtype=f.dtype)
    if reduction is None:
        return f, not_nans

    f[nans] = 0
    if reduction == "mean":
        # 2 steps, first, mean by channel (accounting for nans), then by batch
        not_nans = not_nans.sum(dim=1)
        f = torch.where(not_nans > 0, f.sum(dim=1) / not_nans, t_zero)  # channel average

        not_nans = (not_nans > 0).float().sum(dim=0)
        f = torch.where(not_nans > 0, f.sum(dim=0) / not_nans, t_zero)  # batch average

    elif reduction == "sum":
        not_nans = not_nans.sum(dim=[0, 1])
        f = torch.sum(f, dim=[0, 1])  # sum over the batch and channel dims
    elif reduction == "mean_batch":
        not_nans = not_nans.sum(dim=0)
        f = torch.where(not_nans > 0, f.sum(dim=0) / not_nans, t_zero)  # batch average
    elif reduction == "sum_batch":
        not_nans = not_nans.sum(dim=0)
        f = f.sum(dim=0)  # the batch sum
    elif reduction == "mean_channel":
        not_nans = not_nans.sum(dim=1)
        f = torch.where(not_nans > 0, f.sum(dim=1) / not_nans, t_zero)  # channel average
    elif reduction == "sum_channel":
        not_nans = not_nans.sum(dim=1)
        f = f.sum(dim=1)  # the channel sum
    elif reduction is not None:
        raise ValueError(
            f"Unsupported reduction: {reduction}, available options are "
            '["mean", "sum", "mean_batch", "sum_batch", "mean_channel", "sum_channel" "none"].'
        )
    return f, not_nans
