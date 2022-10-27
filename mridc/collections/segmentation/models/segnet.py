# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from abc import ABC
from typing import Any, Tuple

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch import nn

import mridc.collections.common.parts.fft as fft
import mridc.collections.common.parts.utils as utils
import mridc.collections.segmentation.models.base as base_segmentation_models
import mridc.collections.segmentation.models.idslr_base.idslr_block as idslr_block
import mridc.core.classes.common as common_classes
from mridc.collections.reconstruction.models.rim import conv_layers

__all__ = ["SegNet"]


class SegNet(base_segmentation_models.BaseMRIJointReconstructionSegmentationModel, ABC):
    """
    Implementation of the Segmentation Network MRI, as described in, as presented in Sun, L., \
    et al. (2019).

    References
    ----------

    ..

        Sun, L., Fan, Z., Ding, X., Huang, Y., Paisley, J. (2019). Joint CS-MRI Reconstruction and Segmentation with \
        a Unified Deep Network. In: Chung, A., Gee, J., Yushkevich, P., Bao, S. (eds) Information Processing in \
         Medical Imaging. IPMI 2019. Lecture Notes in Computer Science(), vol 11492. Springer, Cham. \
         https://doi.org/10.1007/978-3-030-20351-1_38

    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # init superclass
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.use_reconstruction_module = cfg_dict.get("use_reconstruction_module", True)

        self.fft_centered = cfg_dict.get("fft_centered")
        self.fft_normalization = cfg_dict.get("fft_normalization")
        self.spatial_dims = cfg_dict.get("spatial_dims")
        self.coil_dim = cfg_dict.get("coil_dim")
        self.coil_combination_method = cfg_dict.get("coil_combination_method")

        self.consecutive_slices = cfg_dict.get("consecutive_slices", 1)
        self.dimensionality = cfg_dict.get("dimensionality", 2)
        if self.dimensionality != 2:
            raise NotImplementedError(f"Currently only 2D is supported for segmentation, got {self.dimensionality}D.")

        self.input_channels = cfg_dict.get("input_channels", 2)
        reconstruction_out_chans = cfg_dict.get("reconstruction_module_output_channels", 2)
        segmentation_out_chans = cfg_dict.get("segmentation_module_output_channels", 1)
        chans = cfg_dict.get("channels", 32)
        num_pools = cfg_dict.get("num_pools", 4)
        padding_size = cfg_dict.get("padding_size", 11)
        drop_prob = cfg_dict.get("drop_prob", 0.0)
        normalize = cfg_dict.get("normalize", True)
        self.norm_groups = cfg_dict.get("norm_groups", 2)
        num_cascades = cfg_dict.get("num_cascades", 5)

        self.reconstruction_encoder = nn.ModuleList(
            [
                idslr_block.UnetEncoder(
                    chans=chans,
                    num_pools=num_pools,
                    in_chans=self.input_channels,
                    drop_prob=drop_prob,
                    padding_size=padding_size,
                    normalize=normalize,
                    norm_groups=self.norm_groups,
                )
                for _ in range(num_cascades)
            ]
        )
        self.reconstruction_decoder = nn.ModuleList(
            [
                idslr_block.UnetDecoder(
                    chans=chans,
                    num_pools=num_pools,
                    out_chans=reconstruction_out_chans,
                    drop_prob=drop_prob,
                    padding_size=padding_size,
                    normalize=normalize,
                    norm_groups=self.norm_groups,
                )
                for _ in range(num_cascades)
            ]
        )
        self.segmentation_decoder = nn.ModuleList(
            [
                idslr_block.UnetDecoder(
                    chans=chans,
                    num_pools=num_pools,
                    out_chans=segmentation_out_chans,
                    drop_prob=drop_prob,
                    padding_size=padding_size,
                    normalize=normalize,
                    norm_groups=self.norm_groups,
                )
                for _ in range(num_cascades)
            ]
        )

        self.segmentation_final_layer = torch.nn.Sequential(
            conv_layers.ConvNonlinear(
                segmentation_out_chans * num_cascades,
                segmentation_out_chans,
                conv_dim=cfg_dict.get("segmentation_final_layer_conv_dim", 2),
                kernel_size=cfg_dict.get("segmentation_final_layer_kernel_size", 3),
                dilation=cfg_dict.get("segmentation_final_layer_dilation", 1),
                bias=cfg_dict.get("segmentation_final_layer_bias", False),
                nonlinear=cfg_dict.get("segmentation_final_layer_nonlinear", "relu"),
            )
        )

        self.dc = idslr_block.DC()

        self.reconstruction_module_accumulate_estimates = False

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
        pred_reconstruction: Predicted reconstruction.
            torch.Tensor, shape [batch_size, n_x, n_y, 2]
        pred_segmentation: Predicted segmentation.
            torch.Tensor, shape [batch_size, nr_classes, n_x, n_y]
        """
        if self.consecutive_slices > 1:
            batch, slices = y.shape[0], y.shape[1]
            y = y.reshape(y.shape[0] * y.shape[1], *y.shape[2:])  # type: ignore
            sensitivity_maps = sensitivity_maps.reshape(
                # type: ignore
                sensitivity_maps.shape[0] * sensitivity_maps.shape[1],
                *sensitivity_maps.shape[2:],
            )
            mask = mask.reshape(mask.shape[0] * mask.shape[1], *mask.shape[2:])  # type: ignore

        # In case of deviating number of coils, we need to pad up to maximum number of coils == number of input \
        # channels for the reconstruction module
        num_coils = y.shape[1]
        if num_coils * 2 != self.input_channels:
            num_coils_to_add = (self.input_channels - num_coils * 2) // 2
            dummy_coil_data = torch.zeros_like(torch.movedim(y, self.coil_dim, 0)[0]).unsqueeze(self.coil_dim)
            for _ in range(num_coils_to_add):
                y = torch.cat([y, dummy_coil_data], dim=self.coil_dim)
                sensitivity_maps = torch.cat([sensitivity_maps, dummy_coil_data], dim=self.coil_dim)

        y_prediction = y.clone()
        pred_segmentations = []
        for re, rd, sd in zip(self.reconstruction_encoder, self.reconstruction_decoder, self.segmentation_decoder):
            image_space = fft.ifft2(y_prediction, self.fft_centered, self.fft_normalization, self.spatial_dims)
            output = re(image_space)
            pred_reconstruction, pad_sizes = output[0].copy(), output[2]

            with torch.no_grad():
                _pred_reconstruction = [
                    torch.nn.functional.group_norm(x, num_groups=self.norm_groups) for x in pred_reconstruction
                ]

            pred_segmentations.append(sd(_pred_reconstruction, iscomplex=False, pad_sizes=pad_sizes))

            pred_kspace = fft.fft2(
                image_space - rd(*output), self.fft_centered, self.fft_normalization, self.spatial_dims
            )
            y_prediction = self.dc(pred_kspace, y, mask)

        pred_reconstruction = self.process_intermediate_pred(
            y_prediction, sensitivity_maps, target_reconstruction, do_coil_combination=True
        )

        pred_segmentation = torch.abs(self.segmentation_final_layer(torch.cat(pred_segmentations, dim=1)))
        pred_segmentation = pred_segmentation / torch.max(pred_segmentation)

        if self.consecutive_slices > 1:
            # get batch size and number of slices from y, because if the reconstruction module is used they will not
            # be saved before
            pred_reconstruction = pred_reconstruction.view([batch, slices, *pred_reconstruction.shape[1:]])
            pred_segmentation = pred_segmentation.view([batch, slices, *pred_segmentation.shape[1:]])

        return pred_reconstruction, pred_segmentation

    def process_intermediate_pred(self, pred, sensitivity_maps, target, do_coil_combination=False):
        """
        Process the intermediate prediction.

        Parameters
        ----------
        pred: Intermediate prediction.
            torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
        sensitivity_maps: Coil sensitivity maps.
            torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
        target: Target data to crop to size.
            torch.Tensor, shape [batch_size, n_x, n_y, 2]
        do_coil_combination: Whether to do coil combination.
            bool, default False

        Returns
        -------
        pred: torch.Tensor, shape [batch_size, n_x, n_y, 2]
            Processed prediction.
        """
        # Take the last time step of the eta
        if do_coil_combination:
            pred = fft.ifft2(
                pred, centered=self.fft_centered, normalization=self.fft_normalization, spatial_dims=self.spatial_dims
            )
            pred = utils.coil_combination(
                pred, sensitivity_maps, method=self.coil_combination_method, dim=self.coil_dim
            )
        pred = torch.view_as_complex(pred)
        if target.shape[-1] == 2:
            target = torch.view_as_complex(target)
        _, pred = utils.center_crop_to_smallest(target, pred)
        return pred
