# coding=utf-8
__author__ = "Dimitrios Karkalousos"

import pytest
import torch
from omegaconf import OmegaConf

from mridc.collections.reconstruction.data.subsample import RandomMaskFunc
from mridc.collections.reconstruction.models.xpdnet import XPDNet
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
                "num_iter": 20,
                "use_primal_only": True,
                "kspace_model_architecture": "CONV",
                "dual_conv_hidden_channels": 16,
                "dual_conv_num_dubs": 2,
                "dual_conv_batchnorm": False,
                "image_model_architecture": "MWCNN",
                "mwcnn_hidden_channels": 16,
                "mwcnn_num_scales": 2,
                "mwcnn_bias": True,
                "mwcnn_batchnorm": False,
                "normalize_image": False,
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
                "num_primal": 5,
                "num_dual": 5,
                "num_iter": 20,
                "use_primal_only": True,
                "kspace_model_architecture": "CONV",
                "dual_conv_hidden_channels": 16,
                "dual_conv_num_dubs": 2,
                "dual_conv_batchnorm": False,
                "image_model_architecture": "MWCNN",
                "mwcnn_hidden_channels": 16,
                "mwcnn_num_scales": 2,
                "mwcnn_bias": True,
                "mwcnn_batchnorm": False,
                "normalize_image": False,
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
def test_xpdnet(shape, cfg, center_fractions, accelerations):
    """
    Test the XPDNet model.

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

    xpdnet = XPDNet(cfg)

    with torch.no_grad():
        y = xpdnet.forward(output, output, mask, output, target=torch.abs(torch.view_as_complex(output)))

    if y.shape[1:] != x.shape[2:4]:
        raise AssertionError
