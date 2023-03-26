# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from abc import ABC

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch import nn

import mridc.collections.reconstruction.nn.base as base_models
import mridc.core.classes.common as common_classes
from mridc.collections.common.parts import fft, utils
from mridc.collections.reconstruction.nn.resnet_base.resnet_block import ConjugateGradient, ResidualNetwork

__all__ = ["ProximalGradient"]


class ProximalGradient(base_models.BaseMRIReconstructionModel, ABC):
    """
    Implementation of the Proximal/Conjugate Gradient, according to [1].

    References
    ----------
    [1] Yaman, B, Hosseini, SAH, Moeller, S, Ellermann, J, Uğurbil, K, Akçakaya, M. Self-supervised learning of
        physics-guided reconstruction neural networks without fully sampled reference data. Magn Reson Med. 2020; 84:
        3172– 3191. https://doi.org/10.1002/mrm.28378

    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.mu = nn.Parameter(torch.Tensor([cfg_dict.get("penalization_weight")]), requires_grad=True)
        self.dc_block = ConjugateGradient(
            cfg_dict.get("conjugate_gradient_iterations", 10),
            self.mu,
            self.fft_centered,
            self.fft_normalization,
            self.spatial_dims,
            self.coil_dim,
            self.coil_combination_method,
        )

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
            method=self.coil_combination_method,
            dim=self.coil_dim,
        )
        return torch.view_as_complex(self.dc_block(prediction, sensitivity_maps, mask))
