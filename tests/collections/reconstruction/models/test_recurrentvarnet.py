# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Parts of the code have been taken from: https://github.com/facebookresearch/fastMRI

import pytest
import torch
from omegaconf import OmegaConf

from mridc.collections.reconstruction.data.subsample import RandomMaskFunc
from mridc.collections.reconstruction.models.rvn import RecurrentVarNet
from mridc.collections.reconstruction.parts import transforms
from tests.collections.reconstruction.fastmri.conftest import create_input


@pytest.mark.parametrize(
    "shape, cfg, center_fractions, accelerations",
    [
        (
            [1, 3, 32, 16, 2],
            {
                "in_channels": 2,
                "recurrent_hidden_channels": 64,
                "recurrent_num_layers": 4,
                "num_steps": 8,
                "no_parameter_sharing": True,
                "learned_initializer": True,
                "initializer_initialization": "sense",
                "initializer_channels": [32, 32, 64, 64],
                "initializer_dilations": [1, 1, 2, 4],
                "initializer_multiscale": 1,
                "use_sens_net": False,
                "coil_combination_method": "SENSE",
                "fft_centered": True,
                "fft_normalization": "ortho",
                "spatial_dims": [-2, -1],
                "coil_dim": 1,
            },
            [0.08],
            [4],
        ),
        (
            [1, 5, 15, 12, 2],
            {
                "in_channels": 2,
                "recurrent_hidden_channels": 64,
                "recurrent_num_layers": 4,
                "num_steps": 8,
                "no_parameter_sharing": False,
                "learned_initializer": True,
                "initializer_initialization": "sense",
                "initializer_channels": [32, 32, 64, 64],
                "initializer_dilations": [1, 1, 2, 4],
                "initializer_multiscale": 1,
                "use_sens_net": False,
                "coil_combination_method": "SENSE",
                "fft_centered": True,
                "fft_normalization": "ortho",
                "spatial_dims": [-2, -1],
                "coil_dim": 1,
            },
            [0.08],
            [4],
        ),
        (
            [1, 8, 13, 18, 2],
            {
                "in_channels": 2,
                "recurrent_hidden_channels": 64,
                "recurrent_num_layers": 4,
                "num_steps": 8,
                "no_parameter_sharing": False,
                "learned_initializer": False,
                "use_sens_net": False,
                "coil_combination_method": "SENSE",
                "fft_centered": True,
                "fft_normalization": "ortho",
                "spatial_dims": [-2, -1],
                "coil_dim": 1,
            },
            [0.08],
            [4],
        ),
        (
            [1, 2, 17, 19, 2],
            {
                "in_channels": 2,
                "recurrent_hidden_channels": 64,
                "recurrent_num_layers": 4,
                "num_steps": 8,
                "no_parameter_sharing": True,
                "learned_initializer": False,
                "use_sens_net": False,
                "coil_combination_method": "SENSE",
                "fft_centered": True,
                "fft_normalization": "ortho",
                "spatial_dims": [-2, -1],
                "coil_dim": 1,
            },
            [0.08],
            [4],
        ),
        (
            [1, 2, 17, 19, 2],
            {
                "in_channels": 2,
                "recurrent_hidden_channels": 64,
                "recurrent_num_layers": 4,
                "num_steps": 18,
                "no_parameter_sharing": True,
                "learned_initializer": False,
                "use_sens_net": False,
                "coil_combination_method": "SENSE",
                "fft_centered": True,
                "fft_normalization": "ortho",
                "spatial_dims": [-2, -1],
                "coil_dim": 1,
            },
            [0.08],
            [4],
        ),
    ],
)
def test_recurrentvarnet(shape, cfg, center_fractions, accelerations):
    """
    Test RecurrentVarNet with different parameters

    Args:
        shape: shape of the input
        cfg: configuration of the model
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

    rvn = RecurrentVarNet(cfg)

    with torch.no_grad():
        y = rvn.forward(output, output, mask, output, eta=x.sum(1), target=torch.abs(torch.view_as_complex(output)))

    if y.shape[1:] != x.shape[2:4]:
        raise AssertionError
