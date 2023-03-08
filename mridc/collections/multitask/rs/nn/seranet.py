# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from abc import ABC
from typing import List, Tuple, Union

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

import mridc.collections.multitask.rs.nn.base as base_rs_models
import mridc.core.classes.common as common_classes
from mridc.collections.common.parts import center_crop_to_smallest, coil_combination_method
from mridc.collections.multitask.rs.nn.seranet_base.convlstm_unet import ConvLSTMNormUnet
from mridc.collections.multitask.rs.nn.seranet_base.recon_block import (
    SERANetReconstructionBlock,
    SERANetRecurrentBlock,
)
from mridc.collections.reconstruction.nn.cascadenet import ccnn_block
from mridc.collections.reconstruction.nn.conv import conv2d
from mridc.collections.reconstruction.nn.unet_base import unet_block
from mridc.collections.segmentation.nn.attention_unet_base.attention_unet_block import AttentionGate

__all__ = ["SERANet"]


class SERANet(base_rs_models.BaseMRIReconstructionSegmentationModel, ABC):  # type: ignore
    """
    Implementation of the End-to-End Recurrent Attention Network as presented in [1].

    References
    ----------
    .. [1] Huang, Q., Chen, X., Metaxas, D., Nadar, M.S. (2019). Brain Segmentation from k-Space with End-to-End
         Recurrent Attention Network. In: , et al. Medical Image Computing and Computer Assisted Intervention –
         MICCAI 2019. Lecture Notes in Computer Science(), vol 11766. Springer, Cham.
         https://doi.org/10.1007/978-3-030-32248-9_31
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.input_channels = cfg_dict.get("input_channels", 2)
        if self.input_channels == 0:
            raise ValueError("Segmentation module input channels cannot be 0.")
        if self.input_channels > 2:
            raise ValueError(f"Segmentation module input channels must be either 1 or 2. Found: {self.input_channels}")
        self.consecutive_slices = cfg_dict.get("consecutive_slices", 1)

        reconstruction_module = cfg_dict.get("reconstruction_module", "unet")
        reconstruction_module_output_channels = cfg_dict.get("reconstruction_module_output_channels", 1)
        if reconstruction_module.lower() == "unet":
            regularizer = unet_block.Unet(
                in_chans=self.input_channels,
                out_chans=reconstruction_module_output_channels,
                chans=cfg_dict.get("reconstruction_module_channels", 64),
                num_pool_layers=cfg_dict.get("reconstruction_module_pooling_layers", 2),
                drop_prob=cfg_dict.get("reconstruction_module_dropout", 0.0),
            )
        elif reconstruction_module.lower() == "cascadenet":
            regularizer = torch.nn.ModuleList(
                [
                    ccnn_block.CascadeNetBlock(
                        conv2d.Conv2d(
                            in_channels=self.input_channels,
                            out_channels=reconstruction_module_output_channels,
                            hidden_channels=cfg_dict.get("reconstruction_module_hidden_channels", 64),
                            n_convs=cfg_dict.get("reconstruction_module_n_convs", 2),
                            batchnorm=cfg_dict.get("reconstruction_module_batchnorm", True),
                        ),
                        fft_centered=self.fft_centered,
                        fft_normalization=self.fft_normalization,
                        spatial_dims=self.spatial_dims,
                        coil_dim=self.coil_dim if self.consecutive_slices == 1 else self.coil_dim - 1,
                        no_dc=True,
                    )
                    for _ in range(cfg_dict.get("reconstruction_module_num_cascades", 5))
                ]
            )
        else:
            raise ValueError(f"Unknown reconstruction module: {reconstruction_module} for SERANet")

        self.reconstruction_module = SERANetReconstructionBlock(
            num_reconstruction_blocks=cfg_dict.get("reconstruction_module_num_blocks", 3),
            reconstruction_model=regularizer,
            fft_centered=self.fft_centered,
            fft_normalization=self.fft_normalization,
            spatial_dims=self.spatial_dims,
            coil_dim=self.coil_dim if self.consecutive_slices == 1 else self.coil_dim - 1,
            coil_combination_method=self.coil_combination_method,
        )
        self.segmentation_module_input_channels = cfg_dict.get("segmentation_module_input_channels", 2)
        segmentation_module_output_channels = cfg_dict.get("segmentation_module_output_channels", 1)
        self.segmentation_module = ConvLSTMNormUnet(
            in_chans=self.segmentation_module_input_channels,
            out_chans=segmentation_module_output_channels,
            chans=cfg_dict.get("segmentation_module_channels", 64),
            num_pools=cfg_dict.get("segmentation_module_pooling_layers", 2),
            drop_prob=cfg_dict.get("segmentation_module_dropout", 0.0),
        )
        self.recurrent_module = SERANetRecurrentBlock(
            num_iterations=cfg_dict.get("recurrent_module_iterations", 3),
            attention_model=AttentionGate(
                in_chans_x=self.segmentation_module_input_channels * 2,
                in_chans_g=segmentation_module_output_channels,
                out_chans=segmentation_module_output_channels,
            ),
            unet_model=ConvLSTMNormUnet(
                in_chans=self.segmentation_module_input_channels * 2,
                out_chans=segmentation_module_output_channels,
                chans=cfg_dict.get("recurrent_module_attention_channels", 64),
                num_pools=cfg_dict.get("recurrent_module_attention_pooling_layers", 2),
                drop_prob=cfg_dict.get("recurrent_module_attention_dropout", 0.0),
            ),
            fft_centered=self.fft_centered,
            fft_normalization=self.fft_normalization,
            spatial_dims=self.spatial_dims,
        )

        self.magnitude_input = cfg_dict.get("magnitude_input", True)
        self.normalize_segmentation_output = cfg_dict.get("normalize_segmentation_output", True)

    @common_classes.typecheck()  # type: ignore
    def forward(  # noqa: W0221
        self,
        y: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
        init_reconstruction_pred: torch.Tensor,
        target_reconstruction: torch.Tensor,
        hx: torch.Tensor = None,  # noqa: W0613
        sigma: float = 1.0,  # noqa: W0613
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
            y = y.reshape(y.shape[0] * y.shape[1], *y.shape[2:])  # type: ignore
            mask = mask.reshape(mask.shape[0] * mask.shape[1], *mask.shape[2:])  # type: ignore
            sensitivity_maps = sensitivity_maps.reshape(  # type: ignore
                sensitivity_maps.shape[0] * sensitivity_maps.shape[1], *sensitivity_maps.shape[2:]  # type: ignore
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

        reconstruction = self.reconstruction_module(init_reconstruction_pred, y, sensitivity_maps, mask)

        if len(reconstruction) > 1:
            pred_reconstruction = reconstruction[-2]
        else:
            pred_reconstruction = reconstruction[-1]

        with torch.no_grad():
            segmentation = torch.nn.functional.group_norm(reconstruction[-1], num_groups=1)

        if segmentation.shape[-1] == 2:
            segmentation = torch.abs(torch.view_as_complex(segmentation))

        # In case of deviating number of coils, we need to pad up to maximum number of coils == number of input \
        # channels for the reconstruction module
        num_coils = segmentation.shape[1]
        if num_coils != self.segmentation_module_input_channels:
            num_coils_to_add = self.segmentation_module_input_channels - num_coils
            dummy_segmentation_coil_data = torch.zeros_like(
                torch.movedim(segmentation, self.coil_dim, 0)[0]
            ).unsqueeze(self.coil_dim)
            dummy_coil_data = torch.zeros_like(torch.movedim(pred_reconstruction, self.coil_dim, 0)[0]).unsqueeze(
                self.coil_dim
            )
            for _ in range(num_coils_to_add):
                segmentation = torch.cat([segmentation, dummy_segmentation_coil_data], dim=self.coil_dim)
                pred_reconstruction = torch.cat([pred_reconstruction, dummy_coil_data], dim=self.coil_dim)
                y = torch.cat([y, dummy_coil_data], dim=self.coil_dim)
                sensitivity_maps = torch.cat([sensitivity_maps, dummy_coil_data], dim=self.coil_dim)

        segmentation = self.segmentation_module(segmentation)

        pred_segmentation = self.recurrent_module(pred_reconstruction, segmentation, y, sensitivity_maps, mask)
        pred_segmentation = torch.abs(pred_segmentation)

        if self.normalize_segmentation_output:
            pred_segmentation = pred_segmentation / torch.max(pred_segmentation)

        pred_reconstruction = coil_combination_method(
            pred_reconstruction,
            sensitivity_maps,
            method=self.coil_combination_method,
            dim=self.coil_dim if self.consecutive_slices == 1 else self.coil_dim - 1,
        )
        pred_reconstruction = torch.view_as_complex(pred_reconstruction)
        if target_reconstruction.shape[-1] == 2:
            target_reconstruction = torch.view_as_complex(target_reconstruction)
        _, pred_reconstruction = center_crop_to_smallest(target_reconstruction, pred_reconstruction)

        if self.consecutive_slices > 1:
            pred_reconstruction = pred_reconstruction.view([batch, slices, *pred_reconstruction.shape[1:]])
            pred_segmentation = pred_segmentation.view([batch, slices, *pred_segmentation.shape[1:]])

        return pred_reconstruction, pred_segmentation
