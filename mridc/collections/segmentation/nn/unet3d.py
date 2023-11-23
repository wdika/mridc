# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from abc import ABC

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

import mridc.collections.segmentation.nn.base as base_segmentation_models
import mridc.core.classes.common as common_classes
from mridc.collections.segmentation.nn.unet3d_base import unet3d_block

__all__ = ["Segmentation3DUNet"]


class Segmentation3DUNet(base_segmentation_models.BaseMRISegmentationModel, ABC):  # type: ignore
    """
    Implementation of the (3D) UNet for MRI segmentation, as presented in [1].

    References
    ----------
    .. [1] O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image
        segmentation. In International Conference on Medical image computing and computer-assisted intervention, pages
        234–241. Springer, 2015.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.segmentation_module = unet3d_block.UNet3D(
            in_chans=self.input_channels,
            out_chans=cfg_dict.get("segmentation_module_output_channels", 2),
            chans=cfg_dict.get("segmentation_module_channels", 64),
            num_pool_layers=cfg_dict.get("segmentation_module_pooling_layers", 2),
            drop_prob=cfg_dict.get("segmentation_module_dropout", 0.0),
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
                init_reconstruction_pred.shape[0] * init_reconstruction_pred.shape[1],
                *init_reconstruction_pred.shape[2:],  # type: ignore
            )

        if init_reconstruction_pred.shape[-1] == 2:  # type: ignore
            if self.input_channels == 1:
                init_reconstruction_pred = torch.view_as_complex(init_reconstruction_pred)  # .unsqueeze(1)
                if self.magnitude_input:
                    init_reconstruction_pred = torch.abs(init_reconstruction_pred)
            elif self.input_channels == 2:
                if self.magnitude_input:
                    raise ValueError("Magnitude input is not supported for 2-channel input.")
                init_reconstruction_pred = init_reconstruction_pred.permute(0, 3, 1, 2)  # type: ignore
            else:
                raise ValueError(f"The input channels must be either 1 or 2. Found: {self.input_channels}")

        if init_reconstruction_pred.dim() == 3:
            init_reconstruction_pred = init_reconstruction_pred.unsqueeze(0)

        if init_reconstruction_pred.dim() == 4:
            init_reconstruction_pred = init_reconstruction_pred.unsqueeze(0)

        with torch.no_grad():
            init_reconstruction_pred = torch.nn.functional.group_norm(
                init_reconstruction_pred, num_groups=self.input_channels
            )

        pred_segmentation = torch.abs(
            self.segmentation_module(init_reconstruction_pred).permute(0, 2, 1, 3, 4).squeeze(0)
        )

        if self.normalize_segmentation_output:
            pred_segmentation = pred_segmentation / torch.max(pred_segmentation)

        if self.consecutive_slices > 1:
            pred_segmentation = pred_segmentation.reshape(batch * slices, *pred_segmentation.shape[1:])

        return pred_segmentation
