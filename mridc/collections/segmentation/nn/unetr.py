# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from abc import ABC

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

import mridc.collections.segmentation.nn.base as base_segmentation_models
import mridc.core.classes.common as common_classes
from mridc.collections.segmentation.nn.unetr_base.unetr_block import UNETR

__all__ = ["SegmentationUNetR"]


class SegmentationUNetR(base_segmentation_models.BaseMRISegmentationModel, ABC):  # type: ignore
    """
    Implementation of the UNETR for MRI segmentation, as presented in [1].

    References
    ----------
    .. Hatamizadeh A, Tang Y, Nath V, Yang D, Myronenko A, Landman B, Roth HR, Xu D. Unetr: Transformers for 3d medical
        image segmentation. InProceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision 2022
        (pp. 574-584).
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.segmentation_module = UNETR(
            in_channels=self.input_channels,
            out_channels=cfg_dict.get("segmentation_module_output_channels", 2),
            img_size=cfg_dict.get("segmentation_module_img_size", (256, 256)),
            feature_size=cfg_dict.get("segmentation_module_channels", 64),
            hidden_size=cfg_dict.get("segmentation_module_hidden_size", 768),
            mlp_dim=cfg_dict.get("segmentation_module_mlp_dim", 3072),
            num_heads=cfg_dict.get("segmentation_module_num_heads", 12),
            pos_embed=cfg_dict.get("segmentation_module_pos_embed", "conv"),
            norm_name=cfg_dict.get("segmentation_module_norm_name", "instance"),
            conv_block=cfg_dict.get("segmentation_module_conv_block", True),
            res_block=cfg_dict.get("segmentation_module_res_block", True),
            dropout_rate=cfg_dict.get("segmentation_module_dropout", 0.0),
            spatial_dims=cfg_dict.get("dimensionality", 2),
            qkv_bias=cfg_dict.get("segmentation_module_qkv_bias", False),
        )

    @common_classes.typecheck()  # type: ignore
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

        pred_segmentation = torch.abs(self.segmentation_module(init_reconstruction_pred))

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
