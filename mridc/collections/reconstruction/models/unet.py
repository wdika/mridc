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

__all__ = ["UNet"]


class UNet(base_models.BaseMRIReconstructionModel, ABC):
    """
    Implementation of the UNet, as presented in O. Ronneberger, P. Fischer, and Thomas Brox.

    References
    ----------
    ..

        O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. \
         In International Conference on Medical image computing and computer-assisted intervention, pages 234–241.  \
         Springer, 2015.

    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # init superclass
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.fft_centered = cfg_dict.get("fft_centered")
        self.fft_normalization = cfg_dict.get("fft_normalization")
        self.spatial_dims = cfg_dict.get("spatial_dims")
        self.coil_dim = cfg_dict.get("coil_dim")

        self.unet = unet_block.NormUnet(
            chans=cfg_dict.get("channels"),
            num_pools=cfg_dict.get("pooling_layers"),
            padding_size=cfg_dict.get("padding_size"),
            normalize=cfg_dict.get("normalize"),
        )

        self.coil_combination_method = cfg_dict.get("coil_combination_method")

        # initialize weights if not using pretrained unet
        # TODO if not cfg_dict.get("pretrained", False):

        if cfg_dict.get("train_loss_fn") == "ssim":
            self.train_loss_fn = losses.SSIMLoss()
        elif cfg_dict.get("train_loss_fn") == "l1":
            self.train_loss_fn = L1Loss()
        elif cfg_dict.get("train_loss_fn") == "mse":
            self.train_loss_fn = torch.nn.MSELoss()
        else:
            raise ValueError("Unknown loss function: {}".format(cfg_dict.get("train_loss_fn")))
        if cfg_dict.get("val_loss_fn") == "ssim":
            self.val_loss_fn = losses.SSIMLoss()
        elif cfg_dict.get("val_loss_fn") == "l1":
            self.val_loss_fn = L1Loss()
        elif cfg_dict.get("val_loss_fn") == "mse":
            self.val_loss_fn = torch.nn.MSELoss()
        else:
            raise ValueError("Unknown loss function: {}".format(cfg_dict.get("val_loss_fn")))

        self.accumulate_estimates = False

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
        eta = torch.view_as_complex(
            utils.coil_combination(
                fft.ifft2(
                    y, centered=self.fft_centered, normalization=self.fft_normalization, spatial_dims=self.spatial_dims
                ),
                sensitivity_maps,
                method=self.coil_combination_method,
                dim=self.coil_dim,
            )
        )
        _, eta = utils.center_crop_to_smallest(target, eta)
        return torch.view_as_complex(self.unet(torch.view_as_real(eta.unsqueeze(self.coil_dim)))).squeeze(
            self.coil_dim
        )
