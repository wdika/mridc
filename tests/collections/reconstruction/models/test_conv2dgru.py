# coding=utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NKI-AI/direct/blob/main/tests/tests_nn/test_recurrent.py
# Copyright (c) DIRECT Contributors

import pytest
import torch

from mridc.collections.reconstruction.models.recurrentvarnet.conv2gru import Conv2dGRU


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
    "hidden_channels",
    [4, 8],
)
def test_conv2dgru(shape, hidden_channels):
    """
    Test the Conv2dGRU model.

    Args:
        shape (): The shape of the input data.
        hidden_channels (): The number of channels in the hidden state.

    Returns:
        None
    """
    model = Conv2dGRU(shape[1], hidden_channels, shape[1])
    data = create_input(shape).cpu()

    out = model(data, None)[0]

    if list(out.shape) != shape:
        raise AssertionError
