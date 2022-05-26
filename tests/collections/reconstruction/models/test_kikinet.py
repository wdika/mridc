# coding=utf-8
__author__ = "Dimitrios Karkalousos"

import pytest
import torch
from omegaconf import OmegaConf

from mridc.collections.reconstruction.data.subsample import RandomMaskFunc
from mridc.collections.reconstruction.models.kikinet import KIKINet
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
                "num_iter": 2,
                "kspace_model_architecture": "UNET",
                "kspace_unet_num_filters": 16,
                "kspace_unet_num_pool_layers": 2,
                "kspace_unet_dropout_probability": 0.0,
                "kspace_unet_padding_size": 11,
                "kspace_unet_normalize": True,
                "imspace_model_architecture": "UNET",
                "imspace_unet_num_filters": 16,
                "imspace_unet_num_pool_layers": 2,
                "imspace_unet_dropout_probability": 0.0,
                "imspace_unet_padding_size": 11,
                "imspace_unet_normalize": True,
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
                "num_iter": 4,
                "kspace_model_architecture": "UNET",
                "kspace_unet_num_filters": 4,
                "kspace_unet_num_pool_layers": 2,
                "kspace_unet_dropout_probability": 0.0,
                "kspace_unet_padding_size": 11,
                "kspace_unet_normalize": True,
                "imspace_model_architecture": "UNET",
                "imspace_unet_num_filters": 4,
                "imspace_unet_num_pool_layers": 2,
                "imspace_unet_dropout_probability": 0.0,
                "imspace_unet_padding_size": 11,
                "imspace_unet_normalize": True,
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
def test_kikinet(shape, cfg, center_fractions, accelerations):
    """
    Test the KIKINet model.

    Args:
        shape (): The shape of the input data.
        cfg (): The configuration of the model.
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

    kikinet = KIKINet(cfg)

    with torch.no_grad():
        y = kikinet.forward(output, output, mask, output, target=torch.abs(torch.view_as_complex(output)))

    if y.shape[1:] != x.shape[2:4]:
        raise AssertionError
