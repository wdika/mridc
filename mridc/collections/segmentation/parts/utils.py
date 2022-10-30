# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from typing import Optional, Union

import numpy as np
import torch

# Taken and adjusted from:
# https://github.com/Project-MONAI/MONAI/blob/f407fcce1c32c050f327b26a16b7f7bc8a01f593/monai/transforms/utils.py#L166


def rescale_intensities(
    data: Union[np.ndarray, torch.Tensor],
    minv: Optional[float] = 0.0,
    maxv: Optional[float] = 1.0,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Rescale the values of numpy array `arr` to be from `minv` to `maxv`.
    If either `minv` or `maxv` is None, it returns `(a - min_a) / (max_a - min_a)`.

    Parameters
    ----------
    data: numpy array or torch tensor.
    minv: minimum value of the output array.
    maxv: maximum value of the output array.

    Returns
    -------
    Rescaled numpy array or torch tensor.
    """
    mina = data.min()
    maxa = data.max()

    if mina == maxa:
        return data * minv if minv is not None else data

    norm = (data - mina) / (maxa - mina)  # normalize the array first
    if (minv is None) or (maxv is None):
        return norm
    return (norm * (maxv - minv)) + minv  # rescale by minv and maxv, which is the normalized array by default
