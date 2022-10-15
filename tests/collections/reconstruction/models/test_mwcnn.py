# coding=utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NKI-AI/direct/blob/main/tests/tests_nn/test_mwcnn.py
# Copyright (c) DIRECT Contributors

import pytest
import torch
import torch.nn as nn

from mridc.collections.reconstruction.models.mwcnn.mwcnn import MWCNN


def create_input(shape):
    """Create a random input tensor."""
    return torch.rand(shape).float()


@pytest.mark.parametrize(
    "shape",
    [
        [3, 2, 32, 32],
        [3, 2, 20, 34],
    ],
)
@pytest.mark.parametrize(
    "first_conv_hidden_channels",
    [4, 8],
)
@pytest.mark.parametrize(
    "n_scales",
    [2, 3],
)
@pytest.mark.parametrize(
    "bias",
    [True, False],
)
@pytest.mark.parametrize(
    "batchnorm",
    [True, False],
)
@pytest.mark.parametrize(
    "act",
    [nn.ReLU(), nn.PReLU()],
)
def test_mwcnn(shape, first_conv_hidden_channels, n_scales, bias, batchnorm, act):
    """
    Test MWCNN model.

    Args:
        shape (): Shape of input data.
        first_conv_hidden_channels (): Number of channels in first convolutional layer.
        n_scales (): Number of scales.
        bias (): Whether to use bias in convolutional layers.
        batchnorm (): Whether to use batch normalization in convolutional layers.
        act (): Activation function.

    Returns:
        None.
    """
    model = MWCNN(shape[1], first_conv_hidden_channels, n_scales, bias, batchnorm, act)

    data = create_input(shape).cpu()

    out = model(data)

    if list(out.shape) != shape:
        raise AssertionError
