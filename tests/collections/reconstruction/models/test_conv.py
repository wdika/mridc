# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NKI-AI/direct/blob/main/tests/tests_nn/test_conv.py
# Copyright (c) DIRECT Contributors

import pytest
import torch
import torch.nn as nn

from mridc.collections.reconstruction.models.conv.conv2d import Conv2d


def create_input(shape):
    """Create a random input tensor."""
    return torch.rand(shape).float()


@pytest.mark.parametrize(
    "shape",
    [
        [3, 2, 32, 32],
        [3, 2, 16, 16],
    ],
)
@pytest.mark.parametrize(
    "out_channels",
    [3, 5],
)
@pytest.mark.parametrize(
    "hidden_channels",
    [16, 8],
)
@pytest.mark.parametrize(
    "n_convs",
    [2, 4],
)
@pytest.mark.parametrize(
    "act",
    [nn.ReLU(), nn.PReLU()],
)
@pytest.mark.parametrize(
    "batchnorm",
    [True, False],
)
def test_conv(shape, out_channels, hidden_channels, n_convs, act, batchnorm):
    """
    Test the Conv2d class.

    Args:
        shape (): The shape of the input data.
        out_channels (): The number of output channels.
        hidden_channels (): The number of hidden channels.
        n_convs (): The number of convolutions.
        act (): The activation function.
        batchnorm (): Whether to use batch normalization.

    Returns:
        None
    """
    model = Conv2d(shape[1], out_channels, hidden_channels, n_convs, act, batchnorm)

    data = create_input(shape).cpu()

    out = model(data)

    if list(out.shape) != [shape[0]] + [out_channels] + shape[2:]:
        raise AssertionError
