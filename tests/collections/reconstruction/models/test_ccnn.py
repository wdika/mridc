# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Parts of the code have been taken from: https://github.com/facebookresearch/fastMRI

import pytest
import torch
from omegaconf import OmegaConf

from mridc.collections.reconstruction.data.subsample import RandomMaskFunc
from mridc.collections.reconstruction.models.ccnn import CascadeNet
from mridc.collections.reconstruction.parts import transforms
from tests.collections.reconstruction.fastmri.conftest import create_input


@pytest.mark.parametrize(
    "shape, cfg, center_fractions, accelerations",
    [
        (
            [1, 3, 32, 16, 2],
            {
                "num_cascades": 10,
                "hidden_channels": 64,
                "n_convs": 5,
                "batchnorm": False,
                "no_dc": False,
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
                "num_cascades": 10,
                "hidden_channels": 64,
                "n_convs": 5,
                "batchnorm": False,
                "no_dc": True,
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
                "num_cascades": 10,
                "hidden_channels": 64,
                "n_convs": 5,
                "batchnorm": True,
                "no_dc": False,
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
                "num_cascades": 10,
                "hidden_channels": 64,
                "n_convs": 5,
                "batchnorm": True,
                "no_dc": True,
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
def test_ccnn(shape, cfg, center_fractions, accelerations):
    """
    Test CascadeNet with different parameters

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

    ccnn = CascadeNet(cfg)

    with torch.no_grad():
        y = ccnn.forward(output, output, mask, output, target=torch.abs(torch.view_as_complex(output)))

    if y.shape[1:] != x.shape[2:4]:
        raise AssertionError
