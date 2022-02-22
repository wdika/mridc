# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NKI-AI/direct/blob/main/tests/tests_nn/test_didn.py
# Copyright (c) DIRECT Contributors

import pytest
import torch

from mridc.collections.reconstruction.models.didn.didn import DIDN


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
    "n_dubs",
    [3, 4],
)
@pytest.mark.parametrize(
    "num_convs_recon",
    [3, 4],
)
@pytest.mark.parametrize(
    "skip",
    [True, False],
)
def test_didn(shape, out_channels, hidden_channels, n_dubs, num_convs_recon, skip):
    """
    Test the DIDN

    Args:
        shape (): shape of the input
        out_channels (): number of output channels
        hidden_channels (): number of hidden channels
        n_dubs (): number of dubs
        num_convs_recon (): number of convolutions in the reconstruction network
        skip (): whether to use skip connections or not

    Returns:
        None
    """
    model = DIDN(shape[1], out_channels, hidden_channels, n_dubs, num_convs_recon, skip)

    data = create_input(shape).cpu()

    out = model(data)

    if list(out.shape) != [shape[0]] + [out_channels] + shape[2:]:
        raise AssertionError
