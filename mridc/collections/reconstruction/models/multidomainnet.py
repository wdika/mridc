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
import mridc.collections.reconstruction.models.multidomain.multidomain as multidomain_
import mridc.core.classes.common as common_classes

__all__ = ["MultiDomainNet"]


class MultiDomainNet(base_models.BaseMRIReconstructionModel, ABC):
    """Feature-level multi-domain module. Inspired by AIRS Medical submission to the FastMRI 2020 challenge."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # init superclass
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.fft_normalization = cfg_dict.get("fft_normalization")
        self.spatial_dims = cfg_dict.get("spatial_dims")
        self.coil_dim = cfg_dict.get("coil_dim")
        self.num_cascades = cfg_dict.get("num_cascades")

        standardization = cfg_dict["standardization"]
        if standardization:
            self.standardization = multidomain_.StandardizationLayer(self.coil_dim, -1)

        self.unet = multidomain_.MultiDomainUnet2d(
            # if standardization, in_channels is 4 due to standardized input
            in_channels=4 if standardization else 2,
            out_channels=2,
            num_filters=cfg_dict["num_filters"],
            num_pool_layers=cfg_dict["num_pool_layers"],
            dropout_probability=cfg_dict["dropout_probability"],
            fft_centered=self.fft_centered,
            fft_normalization=self.fft_normalization,
            spatial_dims=self.spatial_dims,
            coil_dim=self.coil_dim,
        )

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

    def _compute_model_per_coil(self, model, data):
        """
        Compute the model per coil.

        Parameters
        ----------
        model: torch.nn.Module
            The model to be computed.
        data: torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
            The data to be computed.

        Returns
        -------
        torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
            The computed output.
        """
        output = []
        for idx in range(data.size(self.coil_dim)):
            subselected_data = data.select(self.coil_dim, idx)
            output.append(model(subselected_data))
        output = torch.stack(output, dim=self.coil_dim)
        return output

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
        image = fft.ifft2(
            y, centered=self.fft_centered, normalization=self.fft_normalization, spatial_dims=self.spatial_dims
        )

        if hasattr(self, "standardization"):
            image = self.standardization(image, sensitivity_maps)

        output_image = self._compute_model_per_coil(self.unet, image.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
        output_image = utils.coil_combination(
            output_image, sensitivity_maps, method=self.coil_combination_method, dim=self.coil_dim
        )
        output_image = torch.view_as_complex(output_image)
        _, output_image = utils.center_crop_to_smallest(target, output_image)
        return output_image
