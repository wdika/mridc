# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from abc import ABC

import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer

import mridc.collections.reconstruction.nn.base as base_models
import mridc.core.classes.common as common_classes
from mridc.collections.common.parts import fft, utils

__all__ = ["ZF"]


class ZF(base_models.BaseMRIReconstructionModel, ABC):  # type: ignore
    """
    Zero-Filled reconstruction using either root-sum-of-squares (RSS) or SENSE (SENSitivity Encoding, as presented in
    [1]).

    References
    ----------
    .. [1] Pruessmann KP, Weiger M, Scheidegger MB, Boesiger P. SENSE: Sensitivity encoding for fast MRI. Magn Reson
        Med 1999; 42:952-962.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)

    @common_classes.typecheck()  # type: ignore
    def forward(  # noqa: W0221
        self,
        y: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,  # noqa: W0613
        init_pred: torch.Tensor,  # noqa: W0613
        target: torch.Tensor,  # noqa: W0613
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
        prediction = utils.coil_combination_method(
            fft.ifft2(
                y, centered=self.fft_centered, normalization=self.fft_normalization, spatial_dims=self.spatial_dims
            ),
            sensitivity_maps,
            method=self.coil_combination_method.upper(),
            dim=self.coil_dim,
        )
        prediction = utils.check_stacked_complex(prediction)
        return prediction
