# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from abc import ABC

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

import mridc.collections.common.parts.fft as fft
import mridc.collections.common.parts.utils as utils
import mridc.collections.reconstruction.nn.base as models_base
import mridc.collections.reconstruction.nn.cascadenet.ccnn_block as ccnn_block
import mridc.collections.reconstruction.nn.conv.conv2d as conv2d
import mridc.core.classes.common as common_classes

__all__ = ["CascadeNet"]


class CascadeNet(models_base.BaseMRIReconstructionModel, ABC):  # type: ignore
    """
    Implementation of the Deep Cascade of Convolutional Neural Networks, as presented in [1].

    References
    ----------
    .. [1] Schlemper, J., Caballero, J., Hajnal, J. V., Price, A., & Rueckert, D., A Deep Cascade of Convolutional
        Neural Networks for MR Image Reconstruction. Information Processing in Medical Imaging (IPMI), 2017.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        # Cascades of CascadeCNN blocks
        self.cascades = torch.nn.ModuleList(
            [
                ccnn_block.CascadeNetBlock(
                    conv2d.Conv2d(
                        in_channels=2,
                        out_channels=2,
                        hidden_channels=cfg_dict.get("hidden_channels"),
                        n_convs=cfg_dict.get("n_convs"),
                        batchnorm=cfg_dict.get("batchnorm"),
                    ),
                    fft_centered=self.fft_centered,
                    fft_normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                    coil_dim=self.coil_dim,
                    no_dc=cfg_dict.get("no_dc"),
                )
                for _ in range(cfg_dict.get("num_cascades"))
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
        pred = y.clone()
        for cascade in self.cascades:
            pred = cascade(pred, y, sensitivity_maps, mask)
        pred = torch.view_as_complex(
            utils.coil_combination_method(
                fft.ifft2(
                    pred,
                    centered=self.fft_centered,
                    normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                ),
                sensitivity_maps,
                method=self.coil_combination_method,
                dim=self.coil_dim,
            )
        )
        _, pred = utils.center_crop_to_smallest(target, pred)
        return pred
