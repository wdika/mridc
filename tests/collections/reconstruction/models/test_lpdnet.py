# coding=utf-8
__author__ = "Dimitrios Karkalousos"

import pytest
import torch
from omegaconf import OmegaConf

from mridc.collections.reconstruction.data.subsample import RandomMaskFunc
from mridc.collections.reconstruction.models.lpd import LPDNet
from mridc.collections.reconstruction.parts import transforms


def create_input(shape):
    """Create a random input tensor."""
    return torch.rand(shape).float()


@pytest.mark.parametrize(
    "shape, cfg, center_fractions, accelerations",
    [
        (
            [1, 3, 32, 16, 2],
            {
                "num_primal": 5,
                "num_dual": 5,
                "num_iter": 5,
                "primal_model_architecture": "UNET",
                "primal_unet_num_filters": 16,
                "primal_unet_num_pool_layers": 2,
                "primal_unet_dropout_probability": 0.0,
                "primal_unet_padding_size": 11,
                "primal_unet_normalize": True,
                "dual_model_architecture": "UNET",
                "dual_unet_num_filters": 16,
                "dual_unet_num_pool_layers": 2,
                "dual_unet_dropout_probability": 0.0,
                "dual_unet_padding_size": 11,
                "dual_unet_normalize": True,
                "use_sens_net": False,
                "fft_centered": True,
                "fft_normalization": "ortho",
                "spatial_dims": [-2, -1],
                "coil_dim": 1,
                "coil_combination_method": "SENSE",
            },
            [0.08],
            [4],
        ),
        (
            [1, 3, 32, 16, 2],
            {
                "num_primal": 2,
                "num_dual": 2,
                "num_iter": 2,
                "primal_model_architecture": "UNET",
                "primal_unet_num_filters": 4,
                "primal_unet_num_pool_layers": 4,
                "primal_unet_dropout_probability": 0.0,
                "primal_unet_padding_size": 15,
                "primal_unet_normalize": False,
                "dual_model_architecture": "UNET",
                "dual_unet_num_filters": 4,
                "dual_unet_num_pool_layers": 4,
                "dual_unet_dropout_probability": 0.0,
                "dual_unet_padding_size": 15,
                "dual_unet_normalize": False,
                "use_sens_net": False,
                "fft_centered": True,
                "fft_normalization": "ortho",
                "spatial_dims": [-2, -1],
                "coil_dim": 1,
                "coil_combination_method": "SENSE",
            },
            [0.08],
            [4],
        ),
    ],
)
def test_lpdnet(shape, cfg, center_fractions, accelerations):
    """
    Test the LPDNet model.

    Args:
        shape (): The shape of the input data.
        cfg (): The configuration of the LPDNet model.
        center_fractions (): The center fractions of the subsampling.
        accelerations (): The accelerations of the subsampling.

    Returns:
        None.
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

    lpdnet = LPDNet(cfg)

    with torch.no_grad():
        y = lpdnet.forward(output, output, mask, output, target=torch.abs(torch.view_as_complex(output)))

    if y.shape[1:] != x.shape[2:4]:
        raise AssertionError
