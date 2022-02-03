# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Parts of the code have been taken from https://github.com/facebookresearch/fastMRI

import pytest
import torch
from omegaconf import OmegaConf

from mridc.collections.reconstruction.data.subsample import RandomMaskFunc
from mridc.collections.reconstruction.models.cirim import CIRIM
from mridc.collections.reconstruction.parts import transforms
from tests.collections.reconstruction.fastmri.conftest import create_input


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
