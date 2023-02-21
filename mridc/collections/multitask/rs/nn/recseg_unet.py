# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from abc import ABC
from typing import List, Tuple, Union

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

import mridc.collections.multitask.rs.nn.base as base_rs_models
import mridc.core.classes.common as common_classes
from mridc.collections.reconstruction.nn.unet_base import unet_block

__all__ = ["RecSegUNet"]


class RecSegUNet(base_rs_models.BaseMRIReconstructionSegmentationModel, ABC):  # type: ignore
    """
    Implementation of the Reconstruction Segmentation method using UNets for both the reconstruction and segmentation
    as presented in [1].

    References
    ----------
    .. [1] Sui, B, Lv, J, Tong, X, Li, Y, Wang, C. Simultaneous image reconstruction and lesion segmentation in
    accelerated MRI using multitasking learning. Med Phys. 2021; 48: 7189– 7198. https://doi.org/10.1002/mp.15213
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.input_channels = cfg_dict.get("input_channels", 2)
        if self.input_channels == 0:
            raise ValueError("Segmentation module input channels cannot be 0.")
        if self.input_channels > 2:
            raise ValueError(
                "Segmentation module input channels must be either 1 or 2. Found: {}".format(self.input_channels)
            )

        reconstruction_module_output_channels = cfg_dict.get("reconstruction_module_output_channels", 1)

        self.reconstruction_module = unet_block.Unet(
            in_chans=self.input_channels,
            out_chans=reconstruction_module_output_channels,
            chans=cfg_dict.get("reconstruction_module_channels", 64),
            num_pool_layers=cfg_dict.get("reconstruction_module_pooling_layers", 2),
            drop_prob=cfg_dict.get("reconstruction_module_dropout", 0.0),
        )

        self.segmentation_module = unet_block.Unet(
            in_chans=reconstruction_module_output_channels,
            out_chans=cfg_dict.get("segmentation_module_output_channels", 1),
            chans=cfg_dict.get("segmentation_module_channels", 64),
            num_pool_layers=cfg_dict.get("segmentation_module_pooling_layers", 2),
            drop_prob=cfg_dict.get("segmentation_module_dropout", 0.0),
        )

        self.consecutive_slices = cfg_dict.get("consecutive_slices", 1)
        self.magnitude_input = cfg_dict.get("magnitude_input", True)
        self.normalize_segmentation_output = cfg_dict.get("normalize_segmentation_output", True)

    @common_classes.typecheck()  # type: ignore
    def forward(
        self,
        y: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
        init_reconstruction_pred: torch.Tensor,
        target_reconstruction: torch.Tensor,
        hx: torch.Tensor = None,
        sigma: float = 1.0,
    ) -> Tuple[Union[List, torch.Tensor], torch.Tensor]:
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
        hx : torch.Tensor, optional
            Initial hidden state for the RNN. Default is ``None``.
        sigma : float, optional
            Standard deviation of the noise. Default is ``1.0``.

        Returns
        -------
        Tuple[Union[List, torch.Tensor], torch.Tensor]
            Tuple containing the predicted reconstruction and segmentation.
        """
        if self.consecutive_slices > 1:
            batch, slices = init_reconstruction_pred.shape[:2]
            init_reconstruction_pred = init_reconstruction_pred.reshape(  # type: ignore
                init_reconstruction_pred.shape[0] * init_reconstruction_pred.shape[1],
                *init_reconstruction_pred.shape[2:],  # type: ignore
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

        pred_reconstruction = self.reconstruction_module(init_reconstruction_pred)

        with torch.no_grad():
            _pred_reconstruction = torch.nn.functional.group_norm(pred_reconstruction, num_groups=1)

        pred_segmentation = self.segmentation_module(_pred_reconstruction)

        pred_segmentation = torch.abs(pred_segmentation)

        if self.normalize_segmentation_output:
            pred_segmentation = pred_segmentation / torch.max(pred_segmentation)

        pred_reconstruction = pred_reconstruction.squeeze(1)

        if self.consecutive_slices > 1:
            pred_reconstruction = pred_reconstruction.view([batch, slices, *pred_reconstruction.shape[1:]])
            pred_segmentation = pred_segmentation.view([batch, slices, *pred_segmentation.shape[1:]])

        return pred_reconstruction, pred_segmentation
