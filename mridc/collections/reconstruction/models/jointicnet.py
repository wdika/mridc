# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from abc import ABC

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch.nn import L1Loss

import mridc.collections.common.losses.ssim as losses
import mridc.collections.common.parts.fft as fft
import mridc.collections.common.parts.utils as utils
import mridc.collections.reconstruction.models.base as base_models
import mridc.collections.reconstruction.models.unet_base.unet_block as unet_block
import mridc.core.classes.common as common_classes

__all__ = ["JointICNet"]


class JointICNet(base_models.BaseMRIReconstructionModel, ABC):
    """
    Implementation of the Joint Deep Model-Based MR Image and Coil Sensitivity Reconstruction Network (Joint-ICNet), \
    as presented in Jun, Yohan, et al.

    References
    ----------

    ..

        Jun, Yohan, et al. “Joint Deep Model-Based MR Image and Coil Sensitivity Reconstruction Network (Joint-ICNet) \
         for Fast MRI.” 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), IEEE, 2021, pp. \
          5266–75. DOI.org (Crossref), https://doi.org/10.1109/CVPR46437.2021.00523.

    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # init superclass
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.num_iter = cfg_dict.get("num_iter")
        self.fft_centered = cfg_dict.get("fft_centered")
        self.fft_normalization = cfg_dict.get("fft_normalization")
        self.spatial_dims = cfg_dict.get("spatial_dims")
        self.coil_dim = cfg_dict.get("coil_dim")

        self.kspace_model = unet_block.NormUnet(
            cfg_dict.get("kspace_unet_num_filters"),
            cfg_dict.get("kspace_unet_num_pool_layers"),
            in_chans=2,
            out_chans=2,
            drop_prob=cfg_dict.get("kspace_unet_dropout_probability"),
            padding_size=cfg_dict.get("kspace_unet_padding_size"),
            normalize=cfg_dict.get("kspace_unet_normalize"),
        )

        self.image_model = unet_block.NormUnet(
            cfg_dict.get("imspace_unet_num_filters"),
            cfg_dict.get("imspace_unet_num_pool_layers"),
            in_chans=2,
            out_chans=2,
            drop_prob=cfg_dict.get("imspace_unet_dropout_probability"),
            padding_size=cfg_dict.get("imspace_unet_padding_size"),
            normalize=cfg_dict.get("imspace_unet_normalize"),
        )

        self.sens_net = base_models.BaseSensitivityModel(
            cfg_dict.get("sens_unet_num_filters"),
            cfg_dict.get("sens_unet_num_pool_layers"),
            mask_center=cfg_dict.get("sens_unet_mask_center"),
            fft_centered=self.fft_centered,
            fft_normalization=self.fft_normalization,
            spatial_dims=self.spatial_dims,
            coil_dim=self.coil_dim,
            mask_type=cfg_dict.get("sens_mask_type"),
            drop_prob=cfg_dict.get("sens_unet_dropout_probability"),
            padding_size=cfg_dict.get("sens_unet_padding_size"),
            normalize=cfg_dict.get("sens_unet_normalize"),
        )

        self.conv_out = torch.nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1)

        self.reg_param_I = torch.nn.Parameter(torch.ones(self.num_iter))
        self.reg_param_F = torch.nn.Parameter(torch.ones(self.num_iter))
        self.reg_param_C = torch.nn.Parameter(torch.ones(self.num_iter))

        self.lr_image = torch.nn.Parameter(torch.ones(self.num_iter))
        self.lr_sens = torch.nn.Parameter(torch.ones(self.num_iter))

        self.coil_combination_method = cfg_dict.get("coil_combination_method")

        if cfg_dict.get("train_loss_fn") == "ssim":
            self.train_loss_fn = losses.SSIMLoss()
        elif cfg_dict.get("train_loss_fn") == "l1":
            self.train_loss_fn = L1Loss()
        elif cfg_dict.get("train_loss_fn") == "mse":
            self.train_loss_fn = torch.nn.MSELoss()
        else:
            raise ValueError("Unknown loss function: {}".format(cfg_dict.get("train_loss_fn")))
        if cfg_dict.get("eval_loss_fn") == "ssim":
            self.eval_loss_fn = losses.SSIMLoss()
        elif cfg_dict.get("eval_loss_fn") == "l1":
            self.eval_loss_fn = L1Loss()
        elif cfg_dict.get("eval_loss_fn") == "mse":
            self.eval_loss_fn = torch.nn.MSELoss()
        else:
            raise ValueError("Unknown loss function: {}".format(cfg_dict.get("eval_loss_fn")))

        self.accumulate_estimates = False

    def update_C(self, idx, DC_sens, sensitivity_maps, image, y, mask) -> torch.Tensor:
        """
        Update the coil sensitivity maps.

        .. math::
            C = (1 - 2 * '\'lambda_{k}^{C} * ni_{k}) * C_{k}

            C = 2 * '\'lambda_{k}^{C} * ni_{k} * D_{C}(F^-1(b))

            A(x_{k}) = M * F * (C * x_{k})

            C = 2 * ni_{k} * F^-1(M.T * (M * F * (C * x_{k}) - b)) * x_{k}^*

        Parameters
        ----------
        idx: int
            The current iteration index.
        DC_sens: torch.Tensor [batch_size, num_coils, num_sens_maps, num_rows, num_cols]
            The initial coil sensitivity maps.
        sensitivity_maps: torch.Tensor [batch_size, num_coils, num_sens_maps, num_rows, num_cols]
            The coil sensitivity maps.
        image: torch.Tensor [batch_size, num_coils, num_rows, num_cols]
            The predicted image.
        y: torch.Tensor [batch_size, num_coils, num_rows, num_cols]
            The subsampled k-space data.
        mask: torch.Tensor [batch_size, 1, num_rows, num_cols]
            The subsampled mask.

        Returns
        -------
        sensitivity_maps: torch.Tensor [batch_size, num_coils, num_sens_maps, num_rows, num_cols]
            The updated coil sensitivity maps.
        """
        # (1 - 2 * lambda_{k}^{C} * ni_{k}) * C_{k}
        sense_term_1 = (1 - 2 * self.reg_param_C[idx] * self.lr_sens[idx]) * sensitivity_maps
        # 2 * lambda_{k}^{C} * ni_{k} * D_{C}(F^-1(b))
        sense_term_2 = 2 * self.reg_param_C[idx] * self.lr_sens[idx] * DC_sens
        # A(x_{k}) = M * F * (C * x_{k})
        sense_term_3_A = fft.fft2(
            utils.complex_mul(image.unsqueeze(self.coil_dim), sensitivity_maps),
            centered=self.fft_centered,
            normalization=self.fft_normalization,
            spatial_dims=self.spatial_dims,
        )
        sense_term_3_A = torch.where(mask == 0, torch.tensor([0.0], dtype=y.dtype).to(y.device), sense_term_3_A)
        # 2 * ni_{k} * F^-1(M.T * (M * F * (C * x_{k}) - b)) * x_{k}^*
        sense_term_3_mask = torch.where(
            mask == 1,
            torch.tensor([0.0], dtype=y.dtype).to(y.device),
            sense_term_3_A - y,
        )

        sense_term_3_backward = fft.ifft2(
            sense_term_3_mask,
            centered=self.fft_centered,
            normalization=self.fft_normalization,
            spatial_dims=self.spatial_dims,
        )
        sense_term_3 = (
            2 * self.lr_sens[idx] * sense_term_3_backward * utils.complex_conj(image).unsqueeze(self.coil_dim)
        )
        sensitivity_maps = sense_term_1 + sense_term_2 - sense_term_3
        return sensitivity_maps

    def update_X(self, idx, image, sensitivity_maps, y, mask):
        """
        Update the image.

        .. math::
            x_{k} = (1 - 2 * '\'lamdba_{{k}_{I}} * mi_{k} - 2 * '\'lamdba_{{k}_{F}} * mi_{k}) * x_{k}

            x_{k} = 2 * mi_{k} * ('\'lambda_{{k}_{I}} * D_I(x_{k}) + '\'lambda_{{k}_{F}} * F^-1(D_F(f)))

            A(x{k} - b) = M * F * (C * x{k}) - b

            x_{k} = 2 * mi_{k} * A^* * (A(x{k} - b))

        Parameters
        ----------
        idx: int
            The current iteration index.
        image: torch.Tensor [batch_size, num_coils, num_rows, num_cols]
            The predicted image.
        sensitivity_maps: torch.Tensor [batch_size, num_coils, num_sens_maps, num_rows, num_cols]
            The coil sensitivity maps.
        y: torch.Tensor [batch_size, num_coils, num_rows, num_cols]
            The subsampled k-space data.
        mask: torch.Tensor [batch_size, 1, num_rows, num_cols]
            The subsampled mask.

        Returns
        -------
        image: torch.Tensor [batch_size, num_coils, num_rows, num_cols]
            The updated image.
        """
        # (1 - 2 * lamdba_{k}_{I} * mi_{k} - 2 * lamdba_{k}_{F} * mi_{k}) * x_{k}
        image_term_1 = (
            1 - 2 * self.reg_param_I[idx] * self.lr_image[idx] - 2 * self.reg_param_F[idx] * self.lr_image[idx]
        ) * image
        # D_I(x_{k})
        image_term_2_DI = self.image_model(image.unsqueeze(self.coil_dim)).squeeze(self.coil_dim).contiguous()
        image_term_2_DF = fft.ifft2(
            self.kspace_model(
                fft.fft2(
                    image,
                    centered=self.fft_centered,
                    normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                ).unsqueeze(self.coil_dim)
            )
            .squeeze(self.coil_dim)
            .contiguous(),
            centered=self.fft_centered,
            normalization=self.fft_normalization,
            spatial_dims=self.spatial_dims,
        )
        # 2 * mi_{k} * (lambda_{k}_{I} * D_I(x_{k}) + lambda_{k}_{F} * F^-1(D_F(f)))
        image_term_2 = (
            2
            * self.lr_image[idx]
            * (self.reg_param_I[idx] * image_term_2_DI + self.reg_param_F[idx] * image_term_2_DF)
        )
        # A(x{k}) - b) = M * F * (C * x{k}) - b
        image_term_3_A = fft.fft2(
            utils.complex_mul(image.unsqueeze(self.coil_dim), sensitivity_maps),
            centered=self.fft_centered,
            normalization=self.fft_normalization,
            spatial_dims=self.spatial_dims,
        )
        image_term_3_A = torch.where(mask == 0, torch.tensor([0.0], dtype=y.dtype).to(y.device), image_term_3_A) - y
        # 2 * mi_{k} * A^* * (A(x{k}) - b))
        image_term_3_Aconj = utils.complex_mul(
            fft.ifft2(
                image_term_3_A,
                centered=self.fft_centered,
                normalization=self.fft_normalization,
                spatial_dims=self.spatial_dims,
            ),
            utils.complex_conj(sensitivity_maps),
        ).sum(self.coil_dim)
        image_term_3 = 2 * self.lr_image[idx] * image_term_3_Aconj
        image = image_term_1 + image_term_2 - image_term_3
        return image

    @common_classes.typecheck()
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

        Parameters
        ----------
        y: Subsampled k-space data.
            torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
        sensitivity_maps: Coil sensitivity maps.
            torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
        mask: Sampling mask.
            torch.Tensor, shape [1, 1, n_x, n_y, 1]
        init_pred: Initial prediction.
            torch.Tensor, shape [batch_size, n_x, n_y, 2]
        target: Target data to compute the loss.
            torch.Tensor, shape [batch_size, n_x, n_y, 2]

        Returns
        -------
        pred: list of torch.Tensor, shape [batch_size, n_x, n_y, 2], or  torch.Tensor, shape [batch_size, n_x, n_y, 2]
             If self.accumulate_loss is True, returns a list of all intermediate estimates.
             If False, returns the final estimate.
        """
        DC_sens = self.sens_net(y, mask)
        sensitivity_maps = DC_sens.clone()
        image = utils.complex_mul(
            fft.ifft2(
                y, centered=self.fft_centered, normalization=self.fft_normalization, spatial_dims=self.spatial_dims
            ),
            utils.complex_conj(sensitivity_maps),
        ).sum(self.coil_dim)
        for idx in range(self.num_iter):
            sensitivity_maps = self.update_C(idx, DC_sens, sensitivity_maps, image, y, mask)
            image = self.update_X(idx, image, sensitivity_maps, y, mask)
        image = torch.view_as_complex(image)
        _, image = utils.center_crop_to_smallest(target, image)
        return image
