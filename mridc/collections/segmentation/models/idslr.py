# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from abc import ABC
from typing import Any, Tuple

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch import nn

import mridc.collections.common.parts.fft as fft
import mridc.collections.common.parts.utils as utils
import mridc.collections.segmentation.models.base as base_segmentation_models
import mridc.collections.segmentation.models.idslr_base.idslr_block as idslr_block
import mridc.core.classes.common as common_classes

__all__ = ["IDSLR"]


class IDSLR(base_segmentation_models.BaseMRIJointReconstructionSegmentationModel, ABC):
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

        self.reconstruction_module_input_channels = cfg_dict.get("reconstruction_module_input_channels", 2)
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
                idslr_block.UnetEncoder(
                    chans=chans,
                    num_pools=num_pools,
                    in_chans=self.reconstruction_module_input_channels,
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
                idslr_block.UnetDecoder(
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
        self.seg_head = idslr_block.UnetDecoder(
            chans=chans,
            num_pools=num_pools,
            out_chans=seg_out_chans,
            drop_prob=drop_prob,
            padding_size=padding_size,
            normalize=normalize,
            norm_groups=norm_groups,
        )

        self.dc = nn.ModuleList([idslr_block.DC(soft_dc=soft_dc) for _ in range(num_cascades)])
        self.num_iters = num_iters

        self.reconstruction_module_accumulate_estimates = cfg_dict.get(
            "reconstruction_module_accumulate_estimates", False
        )

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
        if num_coils * 2 != self.reconstruction_module_input_channels:
            num_coils_to_add = (self.reconstruction_module_input_channels - num_coils * 2) // 2
            dummy_coil_data = torch.zeros_like(torch.movedim(y, self.coil_dim, 0)[0]).unsqueeze(self.coil_dim)
            for _ in range(num_coils_to_add):
                y = torch.cat([y, dummy_coil_data], dim=self.coil_dim)
                sensitivity_maps = torch.cat([sensitivity_maps, dummy_coil_data], dim=self.coil_dim)

        preds = []
        for encoder, decoder, dc in zip(self.encoders, self.decoders, self.dc):
            tmp = []
            for _ in range(self.num_iters):
                image_space = fft.ifft2(y, self.fft_centered, self.fft_normalization, self.spatial_dims)
                output = encoder(image_space)
                pred_reconstruction, pad_sizes = output[0].copy(), output[2]
                pred_kspace = fft.fft2(
                    image_space - decoder(*output), self.fft_centered, self.fft_normalization, self.spatial_dims
                )
                y = dc(pred_kspace, y, mask)
                tmp.append(
                    self.process_intermediate_pred(
                        y, sensitivity_maps, target_reconstruction, do_coil_combination=True
                    )
                )
            preds.append(tmp)

        with torch.no_grad():
            pred_reconstruction = [torch.nn.functional.group_norm(x, num_groups=1) for x in pred_reconstruction]

        pred_segmentation = self.seg_head(pred_reconstruction, False, pad_sizes)

        pred_segmentation = torch.abs(pred_segmentation)
        pred_segmentation = pred_segmentation / torch.max(pred_segmentation)

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

    def process_reconstruction_loss(self, target, pred, _loss_fn=None):
        """
        Process the loss.

        Parameters
        ----------
        target: Target data.
            torch.Tensor, shape [batch_size, n_x, n_y, 2]
        pred: Final prediction(s).
            list of torch.Tensor, shape [batch_size, n_x, n_y, 2], or
            torch.Tensor, shape [batch_size, n_x, n_y, 2]
        _loss_fn: Loss function.
            torch.nn.Module, default torch.nn.L1Loss()

        Returns
        -------
        loss: torch.FloatTensor, shape [1]
            If self.accumulate_loss is True, returns an accumulative result of all intermediate losses.
        """
        target = target.to(self.device)
        target = torch.abs(target / torch.max(torch.abs(target)))

        if "ssim" in str(_loss_fn).lower():
            max_value = np.array(torch.max(torch.abs(target)).item()).astype(np.float32)

            def loss_fn(x, y):
                """Calculate the ssim loss."""
                y = torch.abs(y / torch.max(torch.abs(y)))
                return _loss_fn(
                    x,
                    y,
                    data_range=torch.tensor(max_value).unsqueeze(dim=0).to(x.device),
                )

        else:

            def loss_fn(x, y):
                """Calculate other loss."""
                x = torch.abs(x / torch.max(torch.abs(x)))
                y = torch.abs(y / torch.max(torch.abs(y)))
                return _loss_fn(x, y)

        if self.reconstruction_module_accumulate_estimates:
            cascades_loss = []
            for cascade_pred in pred:
                time_steps_loss = [loss_fn(target, time_step_pred) for time_step_pred in cascade_pred]
                _loss = [
                    x * torch.logspace(-1, 0, steps=self.num_iters).to(time_steps_loss[0]) for x in time_steps_loss
                ]
                cascades_loss.append(sum(sum(_loss) / self.num_iters))
            yield sum(list(cascades_loss)) / len(pred)
        else:
            return loss_fn(target, pred)
