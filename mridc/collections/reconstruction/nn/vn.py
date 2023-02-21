# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from abc import ABC

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

import mridc.collections.common.parts.fft as fft
import mridc.collections.common.parts.utils as utils
import mridc.collections.reconstruction.nn.base as base_models
import mridc.collections.reconstruction.nn.unet_base.unet_block as unet_block
import mridc.collections.reconstruction.nn.varnet.vn_block as vn_block
import mridc.core.classes.common as common_classes

__all__ = ["VarNet"]


class VarNet(base_models.BaseMRIReconstructionModel, ABC):  # type: ignore
    """
    Implementation of the End-to-end Variational Network (VN), as presented in [1].

    References
    ----------
    .. [1] Sriram A, Zbontar J, Murrell T, Defazio A, Zitnick CL, Yakubova N, Knoll F, Johnson P. End-to-end
        variational networks for accelerated MRI reconstruction. InInternational Conference on Medical Image Computing
        and Computer-Assisted Intervention 2020 Oct 4 (pp. 64-73). Springer, Cham.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.no_dc = cfg_dict.get("no_dc")
        self.num_cascades = cfg_dict.get("num_cascades")

        # Cascades of VN blocks
        self.cascades = torch.nn.ModuleList(
            [
                vn_block.VarNetBlock(
                    unet_block.NormUnet(
                        chans=cfg_dict.get("channels"),
                        num_pools=cfg_dict.get("pooling_layers"),
                        padding_size=cfg_dict.get("padding_size"),
                        normalize=cfg_dict.get("normalize"),
                    ),
                    fft_centered=self.fft_centered,
                    fft_normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                    coil_dim=self.coil_dim,
                    no_dc=self.no_dc,
                )
                for _ in range(self.num_cascades)
            ]
        )

    @common_classes.typecheck()  # type: ignore
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
        estimation = y.clone()

        for cascade in self.cascades:
            # Forward pass through the cascades
            estimation = cascade(estimation, y, sensitivity_maps, mask)

        estimation = fft.ifft2(
            estimation,
            centered=self.fft_centered,
            normalization=self.fft_normalization,
            spatial_dims=self.spatial_dims,
        )
        estimation = utils.coil_combination_method(
            estimation, sensitivity_maps, method=self.coil_combination_method, dim=self.coil_dim
        )
        estimation = torch.view_as_complex(estimation)
        _, estimation = utils.center_crop_to_smallest(target, estimation)
        return estimation
