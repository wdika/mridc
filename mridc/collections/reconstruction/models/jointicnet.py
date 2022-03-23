# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from abc import ABC
from typing import Dict, Generator, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch.nn import L1Loss

from mridc.collections.common.losses.ssim import SSIMLoss
from mridc.collections.common.parts.fft import fft2c, ifft2c
from mridc.collections.common.parts.utils import complex_conj, complex_mul
from mridc.collections.reconstruction.models.base import BaseMRIReconstructionModel, BaseSensitivityModel
from mridc.collections.reconstruction.models.unet_base.unet_block import NormUnet
from mridc.collections.reconstruction.parts.utils import center_crop_to_smallest
from mridc.collections.common.parts.utils import coil_combination

from mridc.core.classes.common import typecheck

__all__ = ["JointICNet"]


class JointICNet(BaseMRIReconstructionModel, ABC):
    """
    Joint Deep Model-Based MR Image and Coil Sensitivity Reconstruction Network (Joint-ICNet) implementation as
    presented in [1]_.

    References
    ----------
    .. [1] Jun, Yohan, et al. “Joint Deep Model-Based MR Image and Coil Sensitivity Reconstruction Network
    (Joint-ICNet) for Fast MRI.” 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),
    IEEE, 2021, pp. 5266–75. DOI.org (Crossref), https://doi.org/10.1109/CVPR46437.2021.00523.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # init superclass
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.num_iter = cfg_dict.get("num_iter")
        self.fft_type = cfg_dict.get("fft_type")

        self.kspace_model = NormUnet(
            cfg_dict.get("kspace_unet_num_filters"),
            cfg_dict.get("kspace_unet_num_pool_layers"),
            in_chans=2,
            out_chans=2,
            drop_prob=cfg_dict.get("kspace_unet_dropout_probability"),
            padding_size=cfg_dict.get("kspace_unet_padding_size"),
            normalize=cfg_dict.get("kspace_unet_normalize"),
        )

        self.image_model = NormUnet(
            cfg_dict.get("imspace_unet_num_filters"),
            cfg_dict.get("imspace_unet_num_pool_layers"),
            in_chans=2,
            out_chans=2,
            drop_prob=cfg_dict.get("imspace_unet_dropout_probability"),
            padding_size=cfg_dict.get("imspace_unet_padding_size"),
            normalize=cfg_dict.get("imspace_unet_normalize"),
        )

        self.sens_net = BaseSensitivityModel(
            cfg_dict.get("sens_unet_num_filters"),
            cfg_dict.get("sens_unet_num_pool_layers"),
            mask_center=cfg_dict.get("sens_unet_mask_center"),
            fft_type=self.fft_type,
            mask_type=cfg_dict.get("sens_mask_type"),
            drop_prob=cfg_dict.get("sens_unet_dropout_probability"),
            padding_size=cfg_dict.get("sens_unet_padding_size"),
            normalize=cfg_dict.get("sens_unet_normalize"),
        )

        self.conv_out = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1)

        self.reg_param_I = nn.Parameter(torch.ones(self.num_iter))
        self.reg_param_F = nn.Parameter(torch.ones(self.num_iter))
        self.reg_param_C = nn.Parameter(torch.ones(self.num_iter))

        self.lr_image = nn.Parameter(torch.ones(self.num_iter))
        self.lr_sens = nn.Parameter(torch.ones(self.num_iter))

        self._coil_dim = 1
        self._spatial_dims = (2, 3)

        self.train_loss_fn = SSIMLoss() if cfg_dict.get("train_loss_fn") == "ssim" else L1Loss()
        self.eval_loss_fn = SSIMLoss() if cfg_dict.get("eval_loss_fn") == "ssim" else L1Loss()
        self.output_type = cfg_dict.get("output_type")

        self.accumulate_estimates = False

    def update_C(self, idx, DC_sens, sensitivity_maps, image, y, mask):
        """Update the coil sensitivity maps."""
        # (1 - 2 * lambda_{k}^{C} * ni_{k}) * C_{k}
        sense_term_1 = (1 - 2 * self.reg_param_C[idx] * self.lr_sens[idx]) * sensitivity_maps
        # 2 * lambda_{k}^{C} * ni_{k} * D_{C}(F^-1(b))
        sense_term_2 = 2 * self.reg_param_C[idx] * self.lr_sens[idx] * DC_sens
        # A(x_{k}) = M * F * (C * x_{k})
        sense_term_3_A = fft2c(complex_mul(image.unsqueeze(1), sensitivity_maps), fft_type=self.fft_type)
        sense_term_3_A = torch.where(mask == 0, torch.tensor([0.0], dtype=y.dtype).to(y.device), sense_term_3_A)
        # 2 * ni_{k} * F^-1(M.T * (M * F * (C * x_{k}) - b)) * x_{k}^*
        sense_term_3_mask = torch.where(
            mask == 1,
            torch.tensor([0.0], dtype=y.dtype).to(y.device),
            sense_term_3_A - y,
        )

        sense_term_3_backward = ifft2c(sense_term_3_mask, fft_type=self.fft_type)
        sense_term_3 = 2 * self.lr_sens[idx] * sense_term_3_backward * complex_conj(image).unsqueeze(1)
        sensitivity_maps = sense_term_1 + sense_term_2 - sense_term_3
        return sensitivity_maps

    def update_X(self, idx, image, sensitivity_maps, y, mask):
        """Update the image."""
        # (1 - 2 * lamdba_{k}_{I} * mi_{k} - 2 * lamdba_{k}_{F} * mi_{k}) * x_{k}
        image_term_1 = (
            1 - 2 * self.reg_param_I[idx] * self.lr_image[idx] - 2 * self.reg_param_F[idx] * self.lr_image[idx]
        ) * image
        # D_I(x_{k})
        image_term_2_DI = self.image_model(image.unsqueeze(1)).squeeze(1).contiguous()
        image_term_2_DF = ifft2c(
            self.kspace_model(fft2c(image, fft_type=self.fft_type).unsqueeze(1)).squeeze(1).contiguous(),
            fft_type=self.fft_type,
        )
        # 2 * mi_{k} * (lambda_{k}_{I} * D_I(x_{k}) + lambda_{k}_{F} * F^-1(D_F(f)))
        image_term_2 = (
            2
            * self.lr_image[idx]
            * (self.reg_param_I[idx] * image_term_2_DI + self.reg_param_F[idx] * image_term_2_DF)
        )
        # A(x{k}) - b) = M * F * (C * x{k}) - b
        image_term_3_A = fft2c(complex_mul(image.unsqueeze(1), sensitivity_maps), fft_type=self.fft_type)
        image_term_3_A = torch.where(mask == 0, torch.tensor([0.0], dtype=y.dtype).to(y.device), image_term_3_A) - y
        # 2 * mi_{k} * A^* * (A(x{k}) - b))
        image_term_3_Aconj = complex_mul(
            ifft2c(image_term_3_A, fft_type=self.fft_type), complex_conj(sensitivity_maps)
        ).sum(1)
        image_term_3 = 2 * self.lr_image[idx] * image_term_3_Aconj
        image = image_term_1 + image_term_2 - image_term_3
        return image

    @typecheck()
    def forward(
        self,
        y: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
        init_pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the network.
        Args:
            y: torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2], masked kspace data
            sensitivity_maps: torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2], coil sensitivity maps
            mask: torch.Tensor, shape [1, 1, n_x, n_y, 1], sampling mask
            init_pred: torch.Tensor, shape [batch_size, n_x, n_y, 2], initial guess for pred
            target: torch.Tensor, shape [batch_size, n_x, n_y, 2], target data
        Returns:
             Final estimation of the network.
             If self.accumulate_loss is True, returns a list of all intermediate estimates.
             If False, returns the final estimate.
        """
        DC_sens = self.sens_net(y, mask)
        sensitivity_maps = DC_sens.clone()
        image = complex_mul(ifft2c(y, fft_type=self.fft_type), complex_conj(sensitivity_maps)).sum(self._coil_dim)
        for idx in range(self.num_iter):
            sensitivity_maps = self.update_C(idx, DC_sens, sensitivity_maps, image, y, mask)
            image = self.update_X(idx, image, sensitivity_maps, y, mask)
        image = torch.view_as_complex(image)
        _, image = center_crop_to_smallest(target, image)
        return image
