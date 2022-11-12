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
import mridc.collections.reconstruction.models.base as models_base
import mridc.collections.reconstruction.models.cascadenet.ccnn_block as ccnn_block
import mridc.collections.reconstruction.models.conv.conv2d as conv2d
import mridc.core.classes.common as common_classes

__all__ = ["CascadeNet"]


class CascadeNet(models_base.BaseMRIReconstructionModel, ABC):
    """
    Implementation of the Deep Cascade of Convolutional Neural Networks, as presented in Schlemper, J., \
    Caballero, J., Hajnal, J. V., Price, A., & Rueckert, D.

    References
    ----------

    ..

        Schlemper, J., Caballero, J., Hajnal, J. V., Price, A., & Rueckert, D., A Deep Cascade of Convolutional \
        Neural Networks for MR Image Reconstruction. Information Processing in Medical Imaging (IPMI), 2017. \
        Available at: https://arxiv.org/pdf/1703.00555.pdf

    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # init superclass
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.coil_combination_method = cfg_dict.get("coil_combination_method")

        self.fft_centered = cfg_dict.get("fft_centered")
        self.fft_normalization = cfg_dict.get("fft_normalization")
        self.spatial_dims = cfg_dict.get("spatial_dims")
        self.coil_dim = cfg_dict.get("coil_dim")

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
        self.dc_weight = torch.nn.Parameter(torch.ones(1))

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
        pred = y.clone()
        for cascade in self.cascades:
            pred = cascade(pred, y, sensitivity_maps, mask)
        pred = torch.view_as_complex(
            utils.coil_combination(
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
