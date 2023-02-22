# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from abc import ABC

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

import mridc.collections.segmentation.nn.base as base_segmentation_models
import mridc.core.classes.common as common_classes
from mridc.collections.segmentation.nn.dynunet_base.dynunet_block import DynUNet

__all__ = ["SegmentationDYNUNet"]


class SegmentationDYNUNet(base_segmentation_models.BaseMRISegmentationModel, ABC):  # type: ignore
    """
    Implementation of a Dynamic UNet (DynUNet), based on [1].

    References
    ----------
    .. [1] Isensee F, Petersen J, Klein A, Zimmerer D, Jaeger PF, Kohl S, Wasserthal J, Koehler G, Norajitra T, Wirkert
        S, Maier-Hein KH. nnu-net: Self-adapting framework for u-net-based medical image segmentation. arXiv preprint
        arXiv:1809.10486. 2018 Sep 27.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        strides = cfg_dict.get("segmentation_module_strides", (1, 1, 1, 1))
        self.deep_supervision = cfg_dict.get("segmentation_module_deep_supervision", False)
        self.segmentation_module = DynUNet(
            spatial_dims=cfg_dict.get("dimensionality", 2),
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

    @common_classes.typecheck()  # type: ignore
    def forward(  # noqa: R0913
        self,
        y: torch.Tensor,  # noqa: R0913
        sensitivity_maps: torch.Tensor,  # noqa: R0913
        mask: torch.Tensor,  # noqa: R0913
        init_reconstruction_pred: torch.Tensor,
        target_reconstruction: torch.Tensor,  # noqa: R0913
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
                raise ValueError(f"The input channels must be either 1 or 2. Found: {self.input_channels}")
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
