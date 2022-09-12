# coding=utf-8
__author__ = "Dimitrios Karkalousos"

import pytest
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from mridc.collections.reconstruction.data.subsample import RandomMaskFunc
from mridc.collections.reconstruction.models.lpd import LPDNet
from mridc.collections.reconstruction.parts import transforms


def create_input(shape):
    """Create a random input tensor."""
    return torch.rand(shape).float()


@pytest.mark.parametrize(
    "shape, cfg, center_fractions, accelerations, dimensionality, trainer",
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
            2,
            {
                "strategy": "ddp",
                "accelerator": "cpu",
                "num_nodes": 1,
                "max_epochs": 20,
                "precision": 32,
                "enable_checkpointing": False,
                "logger": False,
                "log_every_n_steps": 50,
                "check_val_every_n_epoch": -1,
                "max_steps": -1,
            },
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
            2,
            {
                "strategy": "ddp",
                "accelerator": "cpu",
                "num_nodes": 1,
                "max_epochs": 20,
                "precision": 32,
                "enable_checkpointing": False,
                "logger": False,
                "log_every_n_steps": 50,
                "check_val_every_n_epoch": -1,
                "max_steps": -1,
            },
        ),
    ],
)
def test_lpdnet(shape, cfg, center_fractions, accelerations, dimensionality, trainer):
    """
    Test the LPDNet model.

    Args:
        shape: shape of the input
        cfg: configuration of the model
        center_fractions: center fractions
        accelerations: accelerations
        dimensionality: 2D or 3D inputs
        trainer: trainer configuration

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

    if dimensionality == 3 and shape[1] > 1:
        mask = torch.cat([mask, mask], 1)

    cfg = OmegaConf.create(cfg)
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    trainer = OmegaConf.create(trainer)
    trainer = OmegaConf.create(OmegaConf.to_container(trainer, resolve=True))
    trainer = pl.Trainer(**trainer)

    lpdnet = LPDNet(cfg, trainer=trainer)

    with torch.no_grad():
        y = lpdnet.forward(output, output, mask, output, target=torch.abs(torch.view_as_complex(output)))

    if dimensionality == 3:
        x = x.reshape([x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4], x.shape[5]])

    if y.shape[1:] != x.shape[2:4]:
        raise AssertionError
