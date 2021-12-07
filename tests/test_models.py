# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Parts of the code have been taken from https://github.com/facebookresearch/fastMRI

import numpy as np
import pytest
import torch

from mridc.data import transforms
from mridc.data.subsample import RandomMaskFunc
from mridc.nn import CIRIM, Unet, VarNet
from tests.fastmri.conftest import create_input


@pytest.mark.parametrize(
    "shape, cascades, center_fractions, accelerations",
    [
        ([1, 3, 32, 16, 2], 1, [0.08], [4]),
        ([1, 5, 15, 12, 2], 32, [0.04], [8]),
        ([1, 8, 13, 18, 2], 16, [0.08], [4]),
        ([1, 2, 17, 19, 2], 8, [0.08], [4]),
        ([1, 2, 17, 19, 2], 8, [0.08], [4]),
    ],
)
def test_cirim(shape, cascades, center_fractions, accelerations):
    """
    Test CIRIM with different parameters

    Args:
        shape: shape of the input
        cascades: number of cascades
        center_fractions: center fractions
        accelerations: accelerations

    Returns:
        None
    """
    mask_func = RandomMaskFunc(center_fractions, accelerations)
    x = create_input(shape)

    outputs, masks = [], []
    for i in range(x.shape[0]):
        output, mask, _ = transforms.apply_mask(x[i : i + 1], mask_func, seed=123)
        outputs.append(output)
        masks.append(mask)

    output = torch.cat(outputs)
    mask = torch.cat(masks)

    cirim = CIRIM(
        recurrent_layer="IndRNN",
        num_cascades=cascades,
        no_dc=True,
        keep_eta=True,
        use_sens_net=False,
        sens_mask_type="1D",
    )

    with torch.no_grad():
        y = torch.view_as_complex(
            next(cirim.inference(output, output, mask, accumulate_estimates=True))[0][-1]
        )  # type: ignore

    if y.shape[1:] != x.shape[2:4]:
        raise AssertionError


@pytest.mark.parametrize(
    "shape, out_chans, chans",
    [([1, 1, 32, 16], 5, 1), ([5, 1, 15, 12], 10, 32), ([3, 2, 13, 18], 1, 16), ([1, 2, 17, 19], 3, 8)],
)
def test_unet(shape, out_chans, chans):
    """
    Test Unet with different parameters

    Args:
        shape: shape of the input
        out_chans: number of channels
        chans: number of channels

    Returns:
        None
    """
    x = create_input(shape)

    num_chans = x.shape[1]

    unet = Unet(in_chans=num_chans, out_chans=out_chans, chans=chans, num_pool_layers=2)

    y = unet(x)

    if y.shape[1] != out_chans:
        raise AssertionError


@pytest.mark.parametrize(
    "shape, chans, center_fractions, accelerations, mask_center",
    [
        ([1, 3, 32, 16, 2], 1, [0.08], [4], True),
        ([5, 5, 15, 12, 2], 32, [0.04], [8], True),
        ([3, 8, 13, 18, 2], 16, [0.08], [4], True),
        ([1, 2, 17, 19, 2], 8, [0.08], [4], True),
        ([1, 2, 17, 19, 2], 8, [0.08], [4], False),
    ],
)
def test_varnet(shape, chans, center_fractions, accelerations, mask_center):
    """
    Test VarNet with different parameters

    Args:
        shape: shape of the input
        chans: number of channels
        center_fractions: center fractions
        accelerations: accelerations
        mask_center: whether to mask the center

    Returns:
        None
    """
    mask_func = RandomMaskFunc(center_fractions, accelerations)
    x = create_input(shape)
    outputs, masks = [], []
    for i in range(x.shape[0]):
        output, mask, _ = transforms.apply_mask(x[i : i + 1], mask_func, seed=123)
        outputs.append(output)
        masks.append(mask)

    output = torch.cat(outputs)
    mask = torch.cat(masks)

    varnet = VarNet(num_cascades=2, sens_chans=4, sens_pools=2, chans=chans, pools=2, use_sens_net=True)

    y = varnet(output, np.array([]), mask.byte())

    if y.shape[1:] != x.shape[2:4]:
        raise AssertionError
