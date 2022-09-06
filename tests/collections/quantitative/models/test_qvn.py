# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

import pytest
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from mridc.collections.quantitative.models.qvn import qVarNet
from mridc.collections.reconstruction.data.subsample import RandomMaskFunc
from mridc.collections.reconstruction.parts import transforms
from tests.collections.reconstruction.fastmri.conftest import create_input


@pytest.mark.parametrize(
    "shape, cfg, center_fractions, accelerations, num_TEs, dimensionality, trainer",
    [
        (
            [1, 3, 32, 16, 2],
            {
                "use_reconstruction_module": False,
                "quantitative_module_num_cascades": 8,
                "quantitative_module_channels": 18,
                "quantitative_module_pooling_layers": 4,
                "quantitative_module_in_channels": 8,
                "quantitative_module_out_channels": 8,
                "quantitative_module_padding_size": 11,
                "quantitative_module_normalize": True,
                "quantitative_module_accumulate_estimates": False,
                "quantitative_module_no_dc": False,
                "quantitative_module_signal_forward_model_sequence": "MEGRE",
                "quantitative_module_dimensionality": 2,
                "use_sens_net": False,
                "coil_combination_method": "SENSE",
                "fft_centered": True,
                "fft_normalization": "ortho",
                "spatial_dims": [-2, -1],
                "coil_dim": 2,
                "loss_fn": "ssim",
                "shift_B0_input": False,
            },
            [0.08],
            [4],
            4,
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
            [1, 5, 32, 16, 2],
            {
                "use_reconstruction_module": False,
                "reconstruction_module_num_cascades": 8,
                "reconstruction_module_channels": 18,
                "reconstruction_module_pooling_layers": 4,
                "reconstruction_module_in_channels": 4,
                "reconstruction_module_out_channels": 4,
                "reconstruction_module_padding_size": 11,
                "reconstruction_module_normalize": True,
                "reconstruction_module_accumulate_estimates": False,
                "reconstruction_module_no_dc": False,
                "reconstruction_module_dimensionality": 2,
                "quantitative_module_num_cascades": 8,
                "quantitative_module_channels": 18,
                "quantitative_module_pooling_layers": 4,
                "quantitative_module_in_channels": 8,
                "quantitative_module_out_channels": 8,
                "quantitative_module_padding_size": 11,
                "quantitative_module_normalize": True,
                "quantitative_module_accumulate_estimates": False,
                "quantitative_module_no_dc": False,
                "quantitative_module_signal_forward_model_sequence": "MEGRE",
                "quantitative_module_dimensionality": 2,
                "use_sens_net": False,
                "coil_combination_method": "SENSE",
                "fft_centered": True,
                "fft_normalization": "ortho",
                "spatial_dims": [-2, -1],
                "coil_dim": 2,
                "loss_fn": "ssim",
                "shift_B0_input": False,
            },
            [0.08],
            [4],
            4,
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
            [1, 13, 32, 16, 2],
            {
                "use_reconstruction_module": False,
                "reconstruction_module_num_cascades": 8,
                "reconstruction_module_channels": 18,
                "reconstruction_module_pooling_layers": 4,
                "reconstruction_module_in_channels": 8,
                "reconstruction_module_out_channels": 4,
                "reconstruction_module_padding_size": 11,
                "reconstruction_module_normalize": True,
                "reconstruction_module_accumulate_estimates": False,
                "reconstruction_module_no_dc": False,
                "reconstruction_module_dimensionality": 2,
                "quantitative_module_num_cascades": 8,
                "quantitative_module_channels": 18,
                "quantitative_module_pooling_layers": 4,
                "quantitative_module_in_channels": 8,
                "quantitative_module_out_channels": 8,
                "quantitative_module_padding_size": 11,
                "quantitative_module_normalize": True,
                "quantitative_module_accumulate_estimates": False,
                "quantitative_module_no_dc": False,
                "quantitative_module_signal_forward_model_sequence": "MEGRE_no_phase",
                "quantitative_module_dimensionality": 2,
                "use_sens_net": False,
                "coil_combination_method": "SENSE",
                "fft_centered": True,
                "fft_normalization": "ortho",
                "spatial_dims": [-2, -1],
                "coil_dim": 2,
                "loss_fn": "ssim",
                "shift_B0_input": False,
            },
            [0.08],
            [4],
            4,
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
def test_qvn(shape, cfg, center_fractions, accelerations, num_TEs, dimensionality, trainer):
    """
    Test qVarNet with different parameters

    Parameters
    ----------
    shape : list of int
        Shape of the input data
    cfg : dict
        Dictionary with the parameters of the qRIM model
    center_fractions : list of float
        List of center fractions to test
    accelerations : list of float
        List of acceleration factors to test
    num_TEs : int
        Number of TEs to test
    dimensionality : int
        Dimensionality of the data
    trainer : dict
        Dictionary with the parameters of the trainer
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

    output = torch.stack([output for _ in range(4)], 1)
    qmaps = output.sum(2).sum(-1)

    R2star_map = qmaps[:, 0, ...]
    S0_map = qmaps[:, 1, ...]
    B0_map = qmaps[:, 2, ...]
    phi_map = qmaps[:, 3, ...]

    TEs = torch.rand(num_TEs) * 10

    cfg = OmegaConf.create(cfg)
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    trainer = OmegaConf.create(trainer)
    trainer = OmegaConf.create(OmegaConf.to_container(trainer, resolve=True))
    trainer = pl.Trainer(**trainer)

    qvn = qVarNet(cfg, trainer=trainer)

    with torch.no_grad():
        preds = qvn.forward(
            R2star_map,
            S0_map,
            B0_map,
            phi_map,
            TEs,
            output,
            output[:, 0, ...],
            torch.ones_like(mask),
            mask,
        )

        recon_pred, R2star_map_pred, S0_map_pred, B0_map_pred, phi_map_pred = (
            preds[0],
            preds[1],
            preds[2],
            preds[3],
            preds[4],
        )

        if recon_pred.dim() != 0:
            if isinstance(recon_pred, list):
                recon_pred = recon_pred[-1]
            if isinstance(recon_pred, list):
                recon_pred = recon_pred[-1]
            if recon_pred.shape[1:] != x.shape[2:4]:
                raise AssertionError

        if isinstance(R2star_map_pred, list):
            R2star_map_pred = R2star_map_pred[-1]
            S0_map_pred = S0_map_pred[-1]
            B0_map_pred = B0_map_pred[-1]
            phi_map_pred = phi_map_pred[-1]

        if isinstance(R2star_map_pred, list):
            R2star_map_pred = R2star_map_pred[-1]
            S0_map_pred = S0_map_pred[-1]
            B0_map_pred = B0_map_pred[-1]
            phi_map_pred = phi_map_pred[-1]

    if dimensionality == 3:
        x = x.reshape([x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4], x.shape[5]])

    if R2star_map_pred.shape[1:] != x.shape[2:4]:
        raise AssertionError
    if S0_map_pred.shape[1:] != x.shape[2:4]:
        raise AssertionError
    if B0_map_pred.shape[1:] != x.shape[2:4]:
        raise AssertionError
    if phi_map_pred.shape[1:] != x.shape[2:4]:
        raise AssertionError
