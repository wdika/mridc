# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from abc import ABC
from typing import Any, Tuple

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

import mridc.collections.segmentation.models.base as base_segmentation_models
import mridc.core.classes.common as common_classes
from mridc.collections.segmentation.models.dynunet_base.dynunet_block import DynUNet

__all__ = ["DYNUNet"]


class DYNUNet(base_segmentation_models.BaseMRIJointReconstructionSegmentationModel, ABC):
    """Implementation of the DYNUNet as a module."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # init superclass
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.use_reconstruction_module = cfg_dict.get("use_reconstruction_module", False)

        self.fft_centered = cfg_dict.get("fft_centered")
        self.fft_normalization = cfg_dict.get("fft_normalization")
        self.spatial_dims = cfg_dict.get("spatial_dims")
        self.coil_dim = cfg_dict.get("coil_dim")
        self.coil_combination_method = cfg_dict.get("coil_combination_method")

        self.input_channels = cfg_dict.get("segmentation_module_input_channels", 2)
        if self.input_channels == 0:
            raise ValueError("Segmentation module input channels cannot be 0.")
        if self.input_channels > 2:
            raise ValueError(
                "Segmentation module input channels must be either 1 or 2. Found: {}".format(self.input_channels)
            )

        dimensionality = cfg_dict.get("dimensionality", 2)

        strides = cfg_dict.get("segmentation_module_strides", (1, 1, 1, 1))
        self.deep_supervision = cfg_dict.get("segmentation_module_deep_supervision", False)
        self.segmentation_module = DynUNet(
            spatial_dims=dimensionality,
            in_channels=self.input_channels,
            out_channels=cfg_dict.get("segmentation_module_output_channels", 2),
            kernel_size=cfg_dict.get("segmentation_module_kernel_size", 3),
            strides=strides,
            upsample_kernel_size=strides[1:],
            filters=cfg_dict.get("segmentation_module_channels", 64),
            dropout=cfg_dict.get("segmentation_module_dropout", 0.0),
            norm_name=cfg_dict.get("segmentation_module_norm", "instance"),
            act_name=cfg_dict.get("segmentation_module_activation", "leakyrelu"),
            deep_supervision=self.deep_supervision,
            deep_supr_num=cfg_dict.get("segmentation_module_deep_supervision_levels", 1),
        )

        self.consecutive_slices = cfg_dict.get("consecutive_slices", 1)
        self.magnitude_input = cfg_dict.get("magnitude_input", True)

    @common_classes.typecheck()
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

        pred_segmentation = self.segmentation_module(init_reconstruction_pred)

        if self.deep_supervision:
            if pred_segmentation.dim() == 5:
                # TODO: check if this is correct. They do unbind, but they don't show how they handle the tuples.
                pred_segmentation = torch.sum(pred_segmentation, dim=1)

        pred_segmentation = torch.abs(pred_segmentation)
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

        return torch.empty([]), pred_segmentation