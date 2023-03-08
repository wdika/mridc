# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from abc import ABC

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

import mridc.collections.reconstruction.nn.base as base_models
import mridc.collections.reconstruction.nn.multidomain.multidomain as multidomain_
import mridc.core.classes.common as common_classes
from mridc.collections.common.parts import fft, utils

__all__ = ["MultiDomainNet"]


class MultiDomainNet(base_models.BaseMRIReconstructionModel, ABC):  # type: ignore
    """Feature-level multi-domain module. Inspired by AIRS Medical submission to the FastMRI 2020 challenge."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # init superclass
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

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

    def _compute_model_per_coil(self, model: torch.nn.Module, data: torch.Tensor) -> torch.Tensor:
        """
        Computes the model per coil.

        Parameters
        ----------
        model : torch.nn.Module
            The model to be computed.
        data : torch.Tensor
            The data to be computed. Shape [batch_size, n_coils, n_x, n_y, 2].

        Returns
        -------
        torch.Tensor
            The computed output. Shape [batch_size, n_coils, n_x, n_y, 2].
        """
        output = []
        for idx in range(data.size(self.coil_dim)):
            subselected_data = data.select(self.coil_dim, idx)
            output.append(model(subselected_data))
        output = torch.stack(output, dim=self.coil_dim)
        return output

    @common_classes.typecheck()  # type: ignore
    def forward(  # noqa: W0221
        self,
        y: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,  # noqa: W0613
        init_pred: torch.Tensor,  # noqa: W0613
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the network.

        Parameters
        ----------
        y : torch.Tensor
            Subsampled k-space data. Shape [batch_size, n_coils, n_x, n_y, 2]
        sensitivity_maps : torch.Tensor
            Coil sensitivity maps. Shape [batch_size, n_coils, n_x, n_y, 2]
        mask : torch.Tensor
            Subsampling mask. Shape [1, 1, n_x, n_y, 1]
        init_pred : torch.Tensor
            Initial prediction. Shape [batch_size, n_x, n_y, 2]
        target : torch.Tensor
            Target data to compute the loss. Shape [batch_size, n_x, n_y, 2]

        Returns
        -------
        torch.Tensor
            Reconstructed image. Shape [batch_size, n_x, n_y, 2]
        """
        image = fft.ifft2(
            y, centered=self.fft_centered, normalization=self.fft_normalization, spatial_dims=self.spatial_dims
        )

        if hasattr(self, "standardization"):
            image = self.standardization(image, sensitivity_maps)

        prediction = self._compute_model_per_coil(self.unet, image.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
        prediction = utils.coil_combination_method(
            prediction, sensitivity_maps, method=self.coil_combination_method, dim=self.coil_dim
        )
        prediction = torch.view_as_complex(prediction)
        if target.shape[-1] == 2:
            target = torch.view_as_complex(target)
        _, prediction = utils.center_crop_to_smallest(target, prediction)
        return prediction
