# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

import pytest
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from mridc.collections.reconstruction.data.subsample import RandomMaskFunc
from mridc.collections.reconstruction.parts import transforms
from mridc.collections.segmentation.models.jrscirim import JRSCIRIM
from tests.collections.reconstruction.fastmri.conftest import create_input


@pytest.mark.parametrize(
    "shape, cfg, center_fractions, accelerations, dimensionality, segmentation_classes, trainer",
    [
        (
            [1, 3, 32, 16, 2],
            {
                "use_reconstruction_module": True,
                "reconstruction_module_recurrent_layer": "IndRNN",
                "reconstruction_module_conv_filters": [64, 64, 2],
                "reconstruction_module_conv_kernels": [5, 3, 3],
                "reconstruction_module_conv_dilations": [1, 2, 1],
                "reconstruction_module_conv_bias": [True, True, False],
                "reconstruction_module_recurrent_filters": [64, 64, 0],
                "reconstruction_module_recurrent_kernels": [1, 1, 0],
                "reconstruction_module_recurrent_dilations": [1, 1, 0],
                "reconstruction_module_recurrent_bias": [True, True, False],
                "reconstruction_module_depth": 2,
                "reconstruction_module_conv_dim": 2,
                "reconstruction_module_time_steps": 8,
                "reconstruction_module_num_cascades": 5,
                "reconstruction_module_dimensionality": 2,
                "reconstruction_module_accumulate_estimates": True,
                "reconstruction_module_no_dc": True,
                "reconstruction_module_keep_eta": True,
                "segmentation_module": "UNet",
                "segmentation_module_input_channels": 2,
                "segmentation_module_output_channels": 4,
                "segmentation_module_channels": 64,
                "segmentation_module_pooling_layers": 4,
                "segmentation_module_dropout": 0.0,
                "segmentation_loss_fn": "dice",
                "dice_loss_batched": True,
                "dice_loss_include_background": True,
                "dice_loss_squared_pred": False,
                "dice_loss_normalization": "sigmoid",
                "dice_loss_epsilon": 1e-6,
                "coil_combination_method": "SENSE",
                "consecutive_slices": 5,
                "use_sens_net": False,
                "fft_centered": False,
                "fft_normalization": "backward",
                "spatial_dims": [-2, -1],
                "coil_dim": 1,
            },
            [0.08],
            [4],
            2,
            4,
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
                "use_reconstruction_module": True,
                "reconstruction_module_recurrent_layer": "IndRNN",
                "reconstruction_module_conv_filters": [64, 64, 2],
                "reconstruction_module_conv_kernels": [5, 3, 3],
                "reconstruction_module_conv_dilations": [1, 2, 1],
                "reconstruction_module_conv_bias": [True, True, False],
                "reconstruction_module_recurrent_filters": [64, 64, 0],
                "reconstruction_module_recurrent_kernels": [1, 1, 0],
                "reconstruction_module_recurrent_dilations": [1, 1, 0],
                "reconstruction_module_recurrent_bias": [True, True, False],
                "reconstruction_module_depth": 2,
                "reconstruction_module_conv_dim": 2,
                "reconstruction_module_time_steps": 8,
                "reconstruction_module_num_cascades": 5,
                "reconstruction_module_dimensionality": 2,
                "reconstruction_module_accumulate_estimates": True,
                "reconstruction_module_no_dc": True,
                "reconstruction_module_keep_eta": True,
                "segmentation_module": "UNet",
                "segmentation_module_input_channels": 2,
                "segmentation_module_output_channels": 4,
                "segmentation_module_channels": 64,
                "segmentation_module_pooling_layers": 4,
                "segmentation_module_dropout": 0.0,
                "segmentation_loss_fn": "dice",
                "dice_loss_batched": True,
                "dice_loss_include_background": True,
                "dice_loss_squared_pred": False,
                "dice_loss_normalization": "sigmoid",
                "dice_loss_epsilon": 1e-6,
                "coil_combination_method": "SENSE",
                "consecutive_slices": 1,
                "use_sens_net": False,
                "fft_centered": False,
                "fft_normalization": "backward",
                "spatial_dims": [-2, -1],
                "coil_dim": 1,
            },
            [0.08],
            [4],
            2,
            4,
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
            [1, 6, 3, 32, 16, 2],
            {
                "use_reconstruction_module": True,
                "reconstruction_module_recurrent_layer": "IndRNN",
                "reconstruction_module_conv_filters": [64, 64, 2],
                "reconstruction_module_conv_kernels": [5, 3, 3],
                "reconstruction_module_conv_dilations": [1, 2, 1],
                "reconstruction_module_conv_bias": [True, True, False],
                "reconstruction_module_recurrent_filters": [64, 64, 0],
                "reconstruction_module_recurrent_kernels": [1, 1, 0],
                "reconstruction_module_recurrent_dilations": [1, 1, 0],
                "reconstruction_module_recurrent_bias": [True, True, False],
                "reconstruction_module_depth": 2,
                "reconstruction_module_conv_dim": 3,
                "reconstruction_module_time_steps": 8,
                "reconstruction_module_num_cascades": 5,
                "reconstruction_module_dimensionality": 3,
                "reconstruction_module_accumulate_estimates": True,
                "reconstruction_module_no_dc": True,
                "reconstruction_module_keep_eta": True,
                "segmentation_module": "UNet",
                "segmentation_module_input_channels": 2,
                "segmentation_module_output_channels": 4,
                "segmentation_module_channels": 64,
                "segmentation_module_pooling_layers": 4,
                "segmentation_module_dropout": 0.0,
                "segmentation_loss_fn": "dice",
                "dice_loss_batched": True,
                "dice_loss_include_background": True,
                "dice_loss_squared_pred": False,
                "dice_loss_normalization": "sigmoid",
                "dice_loss_epsilon": 1e-6,
                "coil_combination_method": "SENSE",
                "consecutive_slices": 1,
                "use_sens_net": False,
                "fft_centered": False,
                "fft_normalization": "backward",
                "spatial_dims": [-2, -1],
                "coil_dim": 2,
            },
            [0.08],
            [4],
            3,
            4,
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
            [1, 15, 3, 32, 16, 2],
            {
                "use_reconstruction_module": True,
                "reconstruction_module_recurrent_layer": "IndRNN",
                "reconstruction_module_conv_filters": [64, 64, 2],
                "reconstruction_module_conv_kernels": [5, 3, 3],
                "reconstruction_module_conv_dilations": [1, 2, 1],
                "reconstruction_module_conv_bias": [True, True, False],
                "reconstruction_module_recurrent_filters": [64, 64, 0],
                "reconstruction_module_recurrent_kernels": [1, 1, 0],
                "reconstruction_module_recurrent_dilations": [1, 1, 0],
                "reconstruction_module_recurrent_bias": [True, True, False],
                "reconstruction_module_depth": 2,
                "reconstruction_module_conv_dim": 3,
                "reconstruction_module_time_steps": 8,
                "reconstruction_module_num_cascades": 5,
                "reconstruction_module_dimensionality": 3,
                "reconstruction_module_accumulate_estimates": True,
                "reconstruction_module_no_dc": True,
                "reconstruction_module_keep_eta": True,
                "segmentation_module": "UNet",
                "segmentation_module_input_channels": 2,
                "segmentation_module_output_channels": 4,
                "segmentation_module_channels": 64,
                "segmentation_module_pooling_layers": 4,
                "segmentation_module_dropout": 0.0,
                "segmentation_loss_fn": "dice",
                "dice_loss_batched": True,
                "dice_loss_include_background": True,
                "dice_loss_squared_pred": False,
                "dice_loss_normalization": "sigmoid",
                "dice_loss_epsilon": 1e-6,
                "coil_combination_method": "SENSE",
                "consecutive_slices": 1,
                "use_sens_net": False,
                "fft_centered": False,
                "fft_normalization": "backward",
                "spatial_dims": [-2, -1],
                "coil_dim": 2,
            },
            [0.08],
            [4],
            3,
            4,
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
def test_jrscirim(shape, cfg, center_fractions, accelerations, dimensionality, segmentation_classes, trainer):
    """
    Test Joint Reconstruction & Segmentation with different parameters.

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
    dimensionality : int
        Dimensionality of the data
    segmentation_classes : int
        Number of segmentation classes
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

    coil_dim = cfg.get("coil_dim")
    consecutive_slices = cfg.get("consecutive_slices")
    if consecutive_slices > 1:
        output = torch.cat([output for _ in range(consecutive_slices)], 0)

    cfg = OmegaConf.create(cfg)
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    trainer = OmegaConf.create(trainer)
    trainer = OmegaConf.create(OmegaConf.to_container(trainer, resolve=True))
    trainer = pl.Trainer(**trainer)

    jrscirim = JRSCIRIM(cfg, trainer=trainer)

    with torch.no_grad():
        pred_reconstruction, pred_segmentation = jrscirim.forward(
            output,
            output,
            mask,
            output.sum(coil_dim),
            output.sum(coil_dim),
        )

    if cfg.get("accumulate_estimates"):
        try:
            pred_reconstruction = next(pred_reconstruction)
        except StopIteration:
            pass

    if isinstance(pred_reconstruction, list):
        pred_reconstruction = pred_reconstruction[-1]

    if isinstance(pred_reconstruction, list):
        pred_reconstruction = pred_reconstruction[-1]

    if dimensionality == 3:
        x = x.reshape([x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4], x.shape[5]])

    if consecutive_slices > 1 or dimensionality == 3:
        if pred_reconstruction.shape[1:] != x.shape[2:-1]:
            raise AssertionError
        if output.dim() == 6:
            output = output.reshape(
                [output.shape[0] * output.shape[1], output.shape[2], output.shape[3], output.shape[4], output.shape[5]]
            )
            coil_dim -= 1
        output = torch.view_as_complex(output).sum(coil_dim)
        output = torch.stack([output for _ in range(segmentation_classes)], 1)
        if pred_segmentation.shape != output.shape:
            raise AssertionError
    else:
        if pred_reconstruction.shape[1:] != x.shape[2:-1]:
            raise AssertionError
        output = torch.view_as_complex(torch.stack([output for _ in range(segmentation_classes)], 1).sum(coil_dim + 1))
        if pred_segmentation.shape != output.shape:
            raise AssertionError
