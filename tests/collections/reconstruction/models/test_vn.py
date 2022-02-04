# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Parts of the code have been taken from: https://github.com/facebookresearch/fastMRI

import pytest
import torch
from omegaconf import OmegaConf

from mridc.collections.reconstruction.data.subsample import RandomMaskFunc
from mridc.collections.reconstruction.models.vn import VarNet
from mridc.collections.reconstruction.parts import transforms
from tests.collections.reconstruction.fastmri.conftest import create_input


@pytest.mark.parametrize(
    "shape, cfg, center_fractions, accelerations",
    [
        (
            [1, 3, 32, 16, 2],
            {
                "num_cascades": 12,
                "channels": 14,
                "no_dc": False,
                "pooling_layers": 2,
                "padding_size": 11,
                "normalize": True,
                "use_sens_net": False,
                "output_type": "SENSE",
            },
            [0.08],
            [4],
        ),
        (
            [1, 5, 15, 12, 2],
            {
                "num_cascades": 12,
                "channels": 14,
                "no_dc": True,
                "pooling_layers": 2,
                "padding_size": 11,
                "normalize": True,
                "use_sens_net": False,
                "output_type": "SENSE",
            },
            [0.08],
            [4],
        ),
        (
            [1, 2, 17, 19, 2],
            {
                "num_cascades": 18,
                "channels": 14,
                "no_dc": False,
                "pooling_layers": 2,
                "padding_size": 11,
                "normalize": True,
                "use_sens_net": False,
                "output_type": "SENSE",
            },
            [0.08],
            [4],
        ),
        (
            [1, 2, 17, 19, 2],
            {
                "num_cascades": 2,
                "channels": 14,
                "no_dc": False,
                "pooling_layers": 2,
                "padding_size": 15,
                "normalize": True,
                "use_sens_net": False,
                "output_type": "SENSE",
            },
            [0.08],
            [4],
        ),
    ],
)
def test_vn(shape, cfg, center_fractions, accelerations):
    """
    Test VN with different parameters

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

    vn = VarNet(cfg)

    with torch.no_grad():
        y = vn.forward(output, output, mask, target=torch.abs(torch.view_as_complex(output)))

    assert y.shape[1:] == x.shape[2:4]
