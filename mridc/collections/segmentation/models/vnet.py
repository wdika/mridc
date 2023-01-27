# coding=utf-8
__author__ = "Dimitrios Karkalousos"

import math
from abc import ABC
from typing import List, Tuple

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

import mridc.collections.segmentation.models.base as base_segmentation_models
import mridc.collections.segmentation.models.vnet_base.vnet_block as vnet_block
import mridc.core.classes.common as common_classes

__all__ = ["SegmentationVNet"]


class SegmentationVNet(base_segmentation_models.BaseMRISegmentationModel, ABC):
    """
    Implementation of the V-Net for MRI segmentation, as presented in [1].

    References
    ----------
    .. [1] Fausto Milletari, Nassir Navab, Seyed-Ahmad Ahmadi. V-Net: Fully Convolutional Neural Networks for
        Volumetric Medical Image Segmentation, 2016. https://arxiv.org/abs/1606.04797
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.segmentation_module = vnet_block.VNet(
            in_chans=self.input_channels,
            out_chans=cfg_dict.get("segmentation_module_output_channels", 2),
            act=cfg_dict.get("segmentation_module_activation", "elu"),
            drop_prob=cfg_dict.get("segmentation_module_dropout", 0.0),
            bias=cfg_dict.get("segmentation_module_bias", False),
        )

        self.padding_size = cfg_dict.get("segmentation_module_padding_size", 11)

    @common_classes.typecheck()
    def forward(
        self,
        y: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
        init_reconstruction_pred: torch.Tensor,
        target_reconstruction: torch.Tensor,
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
        init_reconstruction_pred : torch.Tensor
            Initial reconstruction prediction. Shape [batch_size, n_x, n_y, 2]
        target_reconstruction : torch.Tensor
            Target reconstruction. Shape [batch_size, n_x, n_y, 2]

        Returns
        -------
        torch.Tensor
            Predicted segmentation. Shape [batch_size, n_classes, n_x, n_y]
        """
        if self.consecutive_slices > 1:
            batch, slices = init_reconstruction_pred.shape[:2]
            init_reconstruction_pred = init_reconstruction_pred.reshape(  # type: ignore
                # type: ignore
                init_reconstruction_pred.shape[0] * init_reconstruction_pred.shape[1],
                init_reconstruction_pred.shape[2],  # type: ignore
                init_reconstruction_pred.shape[3],  # type: ignore
                init_reconstruction_pred.shape[4],  # type: ignore
            )

        if init_reconstruction_pred.shape[-1] == 2:  # type: ignore
            if self.input_channels == 1:
                init_reconstruction_pred = torch.view_as_complex(init_reconstruction_pred).unsqueeze(1)
                if self.magnitude_input:
                    init_reconstruction_pred = torch.abs(init_reconstruction_pred)
            elif self.input_channels == 2:
                if self.magnitude_input:
                    raise ValueError("Magnitude input is not supported for 2-channel input.")
                init_reconstruction_pred = init_reconstruction_pred.permute(0, 3, 1, 2)  # type: ignore
            else:
                raise ValueError("The input channels must be either 1 or 2. Found: {}".format(self.input_channels))
        else:
            if init_reconstruction_pred.dim() == 3:
                init_reconstruction_pred = init_reconstruction_pred.unsqueeze(1)

        with torch.no_grad():
            init_reconstruction_pred = torch.nn.functional.group_norm(init_reconstruction_pred, num_groups=1)

        init_reconstruction_pred, pad_sizes = self.pad(init_reconstruction_pred)
        pred_segmentation = self.segmentation_module(init_reconstruction_pred)
        pred_segmentation = self.unpad(pred_segmentation, *pad_sizes)
        pred_segmentation = torch.abs(pred_segmentation)

        if self.normalize_segmentation_output:
            pred_segmentation = pred_segmentation / torch.max(pred_segmentation)

        if self.consecutive_slices > 1:
            pred_segmentation = pred_segmentation.view(
                [
                    batch,
                    slices,
                    pred_segmentation.shape[1],
                    pred_segmentation.shape[2],
                    pred_segmentation.shape[3],
                ]
            )

        return pred_segmentation

    def pad(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        """Pad the input with zeros to make it square."""
        _, _, h, w = x.shape
        w_mult = ((w - 1) | self.padding_size) + 1
        h_mult = ((h - 1) | self.padding_size) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        x = torch.nn.functional.pad(x, w_pad + h_pad)
        return x, (h_pad, w_pad, h_mult, w_mult)

    @staticmethod
    def unpad(x: torch.Tensor, h_pad: List[int], w_pad: List[int], h_mult: int, w_mult: int) -> torch.Tensor:
        """Unpad the input."""
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]
