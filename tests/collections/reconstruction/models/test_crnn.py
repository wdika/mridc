# coding=utf-8
__author__ = "Dimitrios Karkalousos"

# Parts of the code have been taken from: https://github.com/facebookresearch/fastMRI

import pytest
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from mridc.collections.common.data.subsample import RandomMaskFunc
from mridc.collections.common.parts import utils
from mridc.collections.reconstruction.nn.crnn import CRNNet
from tests.collections.reconstruction.fastmri.conftest import create_input


@pytest.mark.parametrize(
    "shape, cfg, center_fractions, accelerations, dimensionality, trainer",
    [
        (
            [1, 3, 32, 16, 2],
            {
                "num_iterations": 10,
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
                "train_loss_fn": "l1",
                "val_loss_fn": "l1",
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
            [1, 5, 15, 12, 2],
            {
                "num_iterations": 10,
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
                "train_loss_fn": "mse",
                "val_loss_fn": "mse",
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
            [1, 2, 17, 19, 2],
            {
                "num_iterations": 10,
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
                "train_loss_fn": "ssim",
                "val_loss_fn": "ssim",
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
            [1, 2, 17, 19, 2],
            {
                "num_iterations": 10,
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
                "train_loss_fn": "l1",
                "val_loss_fn": "l1",
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
def test_crnn(shape, cfg, center_fractions, accelerations, dimensionality, trainer):
    """
    Test CRNNet with different parameters

    Args:
        shape: shape of the input
        cfg: configuration of the model
        center_fractions: center fractions
        accelerations: accelerations
        dimensionality: 2D or 3D inputs
        trainer: trainer configuration

    Returns:
        None
    """
    mask_func = RandomMaskFunc(center_fractions, accelerations)
    x = create_input(shape)

    outputs, masks = [], []
    for i in range(x.shape[0]):
        output, mask, _ = utils.apply_mask(x[i : i + 1], mask_func, seed=123)
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

    crnn = CRNNet(cfg, trainer=trainer)

    with torch.no_grad():
        y = crnn.forward(output, output, mask, output, target=torch.abs(torch.view_as_complex(output)))

        try:
            y = next(y)
        except StopIteration:
            pass

        y = y[-1]

    if dimensionality == 3:
        x = x.reshape([x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4], x.shape[5]])

    if y.shape[1:] != x.shape[2:4]:
        raise AssertionError
