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
import mridc.core.classes.common as common_classes

__all__ = ["UNet"]


class UNet(base_models.BaseMRIReconstructionModel, ABC):  # type: ignore
    """
    Implementation of the UNet, as presented in [1].

    References
    ----------
    .. [1] O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image
        segmentation. In International Conference on Medical image computing and computer-assisted intervention, pages
        234–241. Springer, 2015.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.unet = unet_block.NormUnet(
            chans=cfg_dict.get("channels"),
            num_pools=cfg_dict.get("pooling_layers"),
            in_chans=cfg_dict.get("in_channels", 2),
            out_chans=cfg_dict.get("out_channels", 2),
            padding_size=cfg_dict.get("padding_size", 11),
            drop_prob=cfg_dict.get("dropout", 0.0),
            normalize=cfg_dict.get("normalize", True),
            norm_groups=cfg_dict.get("norm_groups", 2),
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
        prediction = torch.view_as_complex(
            utils.coil_combination_method(
                fft.ifft2(
                    y, centered=self.fft_centered, normalization=self.fft_normalization, spatial_dims=self.spatial_dims
                ),
                sensitivity_maps,
                method=self.coil_combination_method,
                dim=self.coil_dim,
            )
        )
        _, prediction = utils.center_crop_to_smallest(target, prediction)
        return torch.view_as_complex(self.unet(torch.view_as_real(prediction.unsqueeze(self.coil_dim)))).squeeze(
            self.coil_dim
        )
