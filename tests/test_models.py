# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Parts of the code have been taken from https://github.com/facebookresearch/fastMRI

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from mridc.collections.reconstruction.data.subsample import RandomMaskFunc
from mridc.collections.reconstruction.models.cirim import CIRIM
from mridc.collections.reconstruction.models.e2evn import VarNet
from mridc.collections.reconstruction.models.unet import Unet
from mridc.collections.reconstruction.parts import transforms
from tests.fastmri.conftest import create_input


@pytest.mark.parametrize(
    "shape, cfg, center_fractions, accelerations",
    [
        (
            [1, 3, 32, 16, 2],
            {
                "recurrent_layer": "IndRNN",
                "conv_filters": [64, 64, 2],
                "conv_kernels": [5, 3, 3],
                "conv_dilations": [1, 2, 1],
                "conv_bias": [True, True, False],
                "recurrent_filters": [64, 64, 0],
                "recurrent_kernels": [1, 1, 0],
                "recurrent_dilations": [1, 1, 0],
                "recurrent_bias": [True, True, False],
                "depth": 2,
                "conv_dim": 2,
                "time_steps": 8,
                "num_cascades": 1,
                "accumulate_estimates": True,
                "no_dc": True,
                "keep_eta": True,
                "use_sens_net": False,
                "output_type": "SENSE",
            },
            [0.08],
            [4],
        ),
        (
            [1, 5, 15, 12, 2],
            {
                "recurrent_layer": "IndRNN",
                "conv_filters": [64, 64, 2],
                "conv_kernels": [5, 3, 3],
                "conv_dilations": [1, 2, 1],
                "conv_bias": [True, True, False],
                "recurrent_filters": [64, 64, 0],
                "recurrent_kernels": [1, 1, 0],
                "recurrent_dilations": [1, 1, 0],
                "recurrent_bias": [True, True, False],
                "depth": 2,
                "conv_dim": 2,
                "time_steps": 8,
                "num_cascades": 32,
                "accumulate_estimates": True,
                "no_dc": True,
                "keep_eta": True,
                "use_sens_net": False,
                "output_type": "SENSE",
            },
            [0.08],
            [4],
        ),
        (
            [1, 8, 13, 18, 2],
            {
                "recurrent_layer": "IndRNN",
                "conv_filters": [64, 64, 2],
                "conv_kernels": [5, 3, 3],
                "conv_dilations": [1, 2, 1],
                "conv_bias": [True, True, False],
                "recurrent_filters": [64, 64, 0],
                "recurrent_kernels": [1, 1, 0],
                "recurrent_dilations": [1, 1, 0],
                "recurrent_bias": [True, True, False],
                "depth": 2,
                "conv_dim": 2,
                "time_steps": 8,
                "num_cascades": 16,
                "accumulate_estimates": True,
                "no_dc": True,
                "keep_eta": True,
                "use_sens_net": False,
                "output_type": "SENSE",
            },
            [0.08],
            [4],
        ),
        (
            [1, 2, 17, 19, 2],
            {
                "recurrent_layer": "IndRNN",
                "conv_filters": [64, 64, 2],
                "conv_kernels": [5, 3, 3],
                "conv_dilations": [1, 2, 1],
                "conv_bias": [True, True, False],
                "recurrent_filters": [64, 64, 0],
                "recurrent_kernels": [1, 1, 0],
                "recurrent_dilations": [1, 1, 0],
                "recurrent_bias": [True, True, False],
                "depth": 2,
                "conv_dim": 2,
                "time_steps": 8,
                "num_cascades": 8,
                "accumulate_estimates": True,
                "no_dc": True,
                "keep_eta": True,
                "use_sens_net": False,
                "output_type": "SENSE",
            },
            [0.08],
            [4],
        ),
        (
            [1, 2, 17, 19, 2],
            {
                "recurrent_layer": "IndRNN",
                "conv_filters": [64, 64, 2],
                "conv_kernels": [5, 3, 3],
                "conv_dilations": [1, 2, 1],
                "conv_bias": [True, True, False],
                "recurrent_filters": [64, 64, 0],
                "recurrent_kernels": [1, 1, 0],
                "recurrent_dilations": [1, 1, 0],
                "recurrent_bias": [True, True, False],
                "depth": 2,
                "conv_dim": 2,
                "time_steps": 8,
                "num_cascades": 8,
                "accumulate_estimates": True,
                "no_dc": True,
                "keep_eta": True,
                "use_sens_net": False,
                "output_type": "SENSE",
            },
            [0.08],
            [4],
        ),
    ],
)
def test_cirim(shape, cfg, center_fractions, accelerations):
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

    cfg = OmegaConf.create(cfg)
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    cirim = CIRIM(cfg)

    with torch.no_grad():
        y = next(cirim.forward(output, output, mask, eta=x.sum(1), target=torch.abs(torch.view_as_complex(output))))[
            -1
        ][
            -1
        ]  # type: ignore

    if y.shape[1:] != x.shape[2:4]:
        print(y.shape, x.shape)
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
