# coding=utf-8
__author__ = "Dimitrios Karkalousos"

import math
from abc import ABC
from typing import Any, Tuple

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from mridc.collections.common.parts.rnn_utils import rnn_weights_init
from mridc.collections.reconstruction.models.rim.rim_block import RIMBlock
from mridc.collections.reconstruction.models.unet_base.unet_block import Unet
from mridc.collections.segmentation.models.base import BaseMRIJointReconstructionSegmentationModel
from mridc.core.classes.common import typecheck

__all__ = ["SegmentationUNet"]


class SegmentationUNet(BaseMRIJointReconstructionSegmentationModel, ABC):
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
        self.coil_combination_method = cfg_dict.get("coil_combination_method")

        self.use_reconstruction_module = False

        self.segmentation_module = Unet(
            in_chans=cfg_dict.get("segmentation_module_input_channels", 2),
            out_chans=cfg_dict.get("segmentation_module_output_channels", 2),
            chans=cfg_dict.get("segmentation_module_channels", 64),
            num_pool_layers=cfg_dict.get("segmentation_module_pooling_layers", 2),
            drop_prob=cfg_dict.get("segmentation_module_dropout", 0.0),
        )

        self.consecutive_slices = cfg_dict.get("consecutive_slices", 1)

    @typecheck()
    def forward(
        self,
        y: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
        init_reconstruction_pred: torch.Tensor,
        target_reconstruction: torch.Tensor,
    ) -> Tuple[Any, Any]:
        """
        Forward pass of the network.

        Parameters
        ----------
        y: Data.
            torch.Tensor, shape [batch_size, n_echoes, n_coils, n_x, n_y, 2]
        sensitivity_maps: Coil sensitivity maps.
            torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
        mask: Sub-sampling mask.
            torch.Tensor, shape [batch_size, 1, n_x, n_y, 2]
        init_reconstruction_pred: Initial reconstruction prediction.
            torch.Tensor, shape [batch_size, 1, n_x, n_y, 2]
        target_reconstruction: Target reconstruction.
            torch.Tensor, shape [batch_size, 1, n_x, n_y, 2]

        Returns
        -------
        pred_reconstruction: void
        pred_segmentation: Predicted segmentation.
            torch.Tensor, shape [batch_size, nr_classes, n_x, n_y]
        """
        if self.consecutive_slices > 1:
            init_reconstruction_pred = init_reconstruction_pred.reshape(  # type: ignore
                init_reconstruction_pred.shape[0] * init_reconstruction_pred.shape[1],  # type: ignore
                init_reconstruction_pred.shape[2],  # type: ignore
                init_reconstruction_pred.shape[3],  # type: ignore
                init_reconstruction_pred.shape[4],  # type: ignore
            )
        if init_reconstruction_pred.shape[-1] == 2:  # type: ignore
            init_reconstruction_pred = init_reconstruction_pred.permute(0, 3, 1, 2)  # type: ignore

        pred_segmentation = self.segmentation_module(init_reconstruction_pred)

        return torch.empty([]), pred_segmentation
