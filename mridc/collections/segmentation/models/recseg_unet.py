# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from abc import ABC
from typing import List, Tuple, Union

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

import mridc.collections.segmentation.models.base as base_segmentation_models
import mridc.core.classes.common as common_classes
from mridc.collections.reconstruction.models.unet_base import unet_block

__all__ = ["RecSegUNet"]


class RecSegUNet(base_segmentation_models.BaseMRIJointReconstructionSegmentationModel, ABC):
    """
    Implementation of the Reconstruction Segmentation method using UNets for both the reconstruction and segmentation
    as presented in Sui, B., et al.

    References
    ----------

    ..

         Sui, B, Lv, J, Tong, X, Li, Y, Wang, C. Simultaneous image reconstruction and lesion segmentation in \
         accelerated MRI using multitasking learning. Med Phys. 2021; 48: 7189– 7198. https://doi.org/10.1002/mp.15213

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

        self.reconstruction_module_accumulate_estimates = False

    @common_classes.typecheck()
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
        y: Undersampled k-space data.
            torch.Tensor, shape [batch_size, n_echoes, n_coils, n_x, n_y, 2]
        sensitivity_maps: Coil sensitivity maps.
            torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
        mask: Sub-sampling mask.
            torch.Tensor, shape [batch_size, 1, n_x, n_y, 2]
        init_reconstruction_pred: Initial reconstruction prediction.
            torch.Tensor, shape [batch_size, 1, n_x, n_y, 2]
        target_reconstruction: Target reconstruction.
            torch.Tensor, shape [batch_size, 1, n_x, n_y, 2]
        hx: Hidden state of the RNN.
            torch.Tensor, shape [batch_size, n_x, n_y, 2]
        sigma: Noise standard deviation.
            float

        Returns
        -------
        pred_reconstruction: Predicted reconstruction.
            torch.Tensor, shape [batch_size, n_x, n_y, 2]
        pred_segmentation: Predicted segmentation.
            torch.Tensor, shape [batch_size, nr_classes, n_x, n_y]
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
