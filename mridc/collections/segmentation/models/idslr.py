# coding=utf-8
__author__ = "Dimitrios Karkalousos, Lysander de Jong"

import math
from abc import ABC
from typing import Any, Tuple

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch import nn

from mridc.collections.common.parts.fft import fft2, ifft2
from mridc.collections.common.parts.utils import coil_combination
from mridc.collections.reconstruction.parts.utils import center_crop_to_smallest
from mridc.collections.segmentation.models.base import BaseMRIJointReconstructionSegmentationModel
from mridc.collections.segmentation.models.idslr_base.idslr_block import DC, UnetDecoder, UnetEncoder
from mridc.core.classes.common import typecheck

__all__ = ["IDSLR"]


class IDSLR(BaseMRIJointReconstructionSegmentationModel, ABC):
    """
    Implementation of the Image domain Deep Structured Low-Rank network, as described in, as presented in \
    Aniket Pramanik, Xiaodong Wu, and Mathews Jacob.

    References
    ----------

    ..

        Aniket Pramanik, Xiaodong Wu, and Mathews Jacob. (2021) ‘Joint Calibrationless Reconstruction and \
        Segmentation of Parallel MRI’. Available at: https://arxiv.org/abs/2105.09220

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

        self.consecutive_slices = cfg_dict.get("consecutive_slices", 1)
        self.dimensionality = cfg_dict.get("dimensionality", 2)
        if self.dimensionality != 2:
            raise NotImplementedError(f"Currently only 2D is supported for segmentation, got {self.dimensionality}D.")

        self.input_channels = cfg_dict.get("reconstruction_module_input_channels", 2)
        out_chans = cfg_dict.get("reconstruction_module_output_channels", 2)
        seg_out_chans = cfg_dict.get("segmentation_module_output_channels", 2)
        num_iters = cfg_dict.get("num_iters", 5)
        soft_dc = cfg_dict.get("soft_dc", True)
        chans = cfg_dict.get("channels", 32)
        num_pools = cfg_dict.get("num_pools", 4)
        padding_size = cfg_dict.get("padding_size", 11)
        drop_prob = cfg_dict.get("drop_prob", 0.0)
        normalize = cfg_dict.get("normalize", True)
        norm_groups = cfg_dict.get("norm_groups", 2)
        num_cascades = cfg_dict.get("num_cascades", 1)

        self.encoders = nn.ModuleList(
            [
                UnetEncoder(
                    chans=chans,
                    num_pools=num_pools,
                    in_chans=self.input_channels,
                    drop_prob=drop_prob,
                    padding_size=padding_size,
                    normalize=normalize,
                    norm_groups=norm_groups,
                )
                for _ in range(num_cascades)
            ]
        )
        self.decoders = nn.ModuleList(
            [
                UnetDecoder(
                    chans=chans,
                    num_pools=num_pools,
                    out_chans=out_chans,
                    drop_prob=drop_prob,
                    padding_size=padding_size,
                    normalize=normalize,
                    norm_groups=norm_groups,
                )
                for _ in range(num_cascades)
            ]
        )
        self.seg_head = UnetDecoder(
            chans=chans,
            num_pools=num_pools,
            out_chans=seg_out_chans,
            drop_prob=drop_prob,
            padding_size=padding_size,
            normalize=normalize,
            norm_groups=norm_groups,
        )

        self.dc = nn.ModuleList([DC(soft_dc=soft_dc) for _ in range(num_cascades)])
        self.num_iters = num_iters

        self.use_reconstruction_module = True
        self.reconstruction_module_accumulate_estimates = cfg_dict.get(
            "reconstruction_module_accumulate_estimates", False
        )

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
        pred_reconstruction: Predicted reconstruction.
            torch.Tensor, shape [batch_size, n_x, n_y, 2]
        pred_segmentation: Predicted segmentation.
            torch.Tensor, shape [batch_size, nr_classes, n_x, n_y]
        """
        if self.consecutive_slices > 1:
            batch, slices = y.shape[0], y.shape[1]
            y = y.reshape(y.shape[0] * y.shape[1], *y.shape[2:])  # type: ignore
            sensitivity_maps = sensitivity_maps.reshape(
                sensitivity_maps.shape[0] * sensitivity_maps.shape[1], *sensitivity_maps.shape[2:]  # type: ignore
            )
            mask = mask.reshape(mask.shape[0] * mask.shape[1], *mask.shape[2:])  # type: ignore

        preds = []
        for encoder, decoder, dc in zip(self.encoders, self.decoders, self.dc):
            tmp = []
            for _ in range(self.num_iters):
                image_space = ifft2(y, self.fft_centered, self.fft_normalization, self.spatial_dims)
                output = encoder(image_space)
                pred_reconstruction, pad_sizes = output[0].copy(), output[2]
                pred_kspace = fft2(
                    image_space - decoder(*output), self.fft_centered, self.fft_normalization, self.spatial_dims
                )
                y = dc(pred_kspace, y, mask)
                tmp.append(
                    self.process_intermediate_pred(
                        y, sensitivity_maps, target_reconstruction, do_coil_combination=True
                    )
                )
            preds.append(tmp)
        pred_segmentation = self.seg_head(pred_reconstruction, False, pad_sizes)
        pred_segmentation = torch.abs(pred_segmentation / torch.abs(torch.max(pred_segmentation)))
        if self.consecutive_slices > 1:
            # get batch size and number of slices from y, because if the reconstruction module is used they will not
            # be saved before
            pred_segmentation = pred_segmentation.view([batch, slices, *pred_segmentation.shape[1:]])

        return preds, pred_segmentation

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
            pred = ifft2(
                pred, centered=self.fft_centered, normalization=self.fft_normalization, spatial_dims=self.spatial_dims
            )
            pred = coil_combination(pred, sensitivity_maps, method=self.coil_combination_method, dim=self.coil_dim)
        pred = torch.view_as_complex(pred)
        if target.shape[-1] == 2:
            target = torch.view_as_complex(target)
        _, pred = center_crop_to_smallest(target, pred)
        return pred