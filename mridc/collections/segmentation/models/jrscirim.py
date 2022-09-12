# coding=utf-8
__author__ = "Dimitrios Karkalousos, Lysander de Jong"

import math
from abc import ABC
from typing import Any, Tuple

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from mridc.collections.common.parts.fft import ifft2
from mridc.collections.common.parts.rnn_utils import rnn_weights_init
from mridc.collections.common.parts.utils import coil_combination
from mridc.collections.reconstruction.models.rim.rim_block import RIMBlock
from mridc.collections.reconstruction.models.unet_base.unet_block import Unet
from mridc.collections.reconstruction.parts.utils import center_crop_to_smallest
from mridc.collections.segmentation.models.attention_unet_base.attention_unet_block import AttentionUnet
from mridc.collections.segmentation.models.base import BaseMRIJointReconstructionSegmentationModel
from mridc.collections.segmentation.models.lambda_unet_base.lambda_unet_block import LambdaBlock
from mridc.collections.segmentation.models.vnet_base.vnet_block import VNet
from mridc.core.classes.common import typecheck

__all__ = ["JRSCIRIM"]


class JRSCIRIM(BaseMRIJointReconstructionSegmentationModel, ABC):
    """
    Implementation of the Joint Reconstruction & Segmentation Cascades of Independently Recurrent Inference Machines,
    as presented in [placeholder].

    References
    ----------

    ..

        Placeholder.

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
        self.input_channels = cfg_dict.get("segmentation_module_input_channels", 2)
        self.magnitude_input = cfg_dict.get("magnitude_input", True)

        self.use_reconstruction_module = True

        reconstruction_module_recurrent_filters = cfg_dict.get("reconstruction_module_recurrent_filters")
        reconstruction_module_num_cascades = cfg_dict.get("reconstruction_module_num_cascades")
        self.reconstruction_module_time_steps = 8 * math.ceil(cfg_dict.get("reconstruction_module_time_steps") / 8)
        self.no_dc = cfg_dict.get("reconstruction_module_no_dc")
        self.keep_eta = cfg_dict.get("reconstruction_module_keep_eta")
        self.reconstruction_module_dimensionality = cfg_dict.get("reconstruction_module_dimensionality")

        reconstruction_module_consecutive_slices = (
            self.consecutive_slices if self.reconstruction_module_dimensionality == 3 else 1
        )

        self.reconstruction_module = torch.nn.ModuleList(
            [
                RIMBlock(
                    recurrent_layer=cfg_dict.get("reconstruction_module_recurrent_layer"),
                    conv_filters=cfg_dict.get("reconstruction_module_conv_filters"),
                    conv_kernels=cfg_dict.get("reconstruction_module_conv_kernels"),
                    conv_dilations=cfg_dict.get("reconstruction_module_conv_dilations"),
                    conv_bias=cfg_dict.get("reconstruction_module_conv_bias"),
                    recurrent_filters=reconstruction_module_recurrent_filters,
                    recurrent_kernels=cfg_dict.get("reconstruction_module_recurrent_kernels"),
                    recurrent_dilations=cfg_dict.get("reconstruction_module_recurrent_dilations"),
                    recurrent_bias=cfg_dict.get("reconstruction_module_recurrent_bias"),
                    depth=cfg_dict.get("reconstruction_module_depth"),
                    time_steps=self.reconstruction_module_time_steps,
                    conv_dim=cfg_dict.get("reconstruction_module_conv_dim"),
                    no_dc=self.no_dc,
                    fft_centered=self.fft_centered,
                    fft_normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                    coil_dim=self.coil_dim - 1,
                    dimensionality=self.reconstruction_module_dimensionality,
                    consecutive_slices=reconstruction_module_consecutive_slices,
                )
                for _ in range(reconstruction_module_num_cascades)
            ]
        )

        # Keep estimation through the cascades if keep_eta is True or re-estimate it if False.
        self.reconstruction_module_keep_eta = cfg_dict.get("reconstruction_module_keep_eta")

        # initialize weights if not using pretrained cirim
        if not cfg_dict.get("pretrained", False):
            std_init_range = 1 / reconstruction_module_recurrent_filters[0] ** 0.5
            self.reconstruction_module.apply(lambda module: rnn_weights_init(module, std_init_range))

        self.dc_weight = torch.nn.Parameter(torch.ones(1))
        self.reconstruction_module_accumulate_estimates = cfg_dict.get("reconstruction_module_accumulate_estimates")

        segmentation_module = cfg_dict.get("segmentation_module")
        if segmentation_module.lower() == "unet":
            self.segmentation_module = Unet(
                in_chans=self.input_channels,
                out_chans=cfg_dict.get("segmentation_module_output_channels", 2),
                chans=cfg_dict.get("segmentation_module_channels", 64),
                num_pool_layers=cfg_dict.get("segmentation_module_pooling_layers", 2),
                drop_prob=cfg_dict.get("segmentation_module_dropout", 0.0),
            )
        elif segmentation_module.lower() == "attentionunet":
            self.segmentation_module = AttentionUnet(
                in_chans=self.input_channels,
                out_chans=cfg_dict.get("segmentation_module_output_channels", 2),
                chans=cfg_dict.get("segmentation_module_channels", 64),
                num_pool_layers=cfg_dict.get("segmentation_module_pooling_layers", 2),
                drop_prob=cfg_dict.get("segmentation_module_dropout", 0.0),
            )
        elif segmentation_module.lower() == "lambdaunet":
            self.segmentation_module = LambdaBlock(
                in_chans=self.input_channels,
                out_chans=cfg_dict.get("segmentation_module_output_channels", 2),
                drop_prob=cfg_dict.get("segmentation_module_dropout", 0.0),
                temporal_kernel=cfg_dict.get("segmentation_module_temporal_kernel", 1),
                num_slices=self.consecutive_slices,
            )
        elif segmentation_module.lower() == "vnet":
            self.segmentation_module = VNet(
                in_chans=self.input_channels,
                out_chans=cfg_dict.get("segmentation_module_output_channels", 2),
                act=cfg_dict.get("segmentation_module_activation", "elu"),
                drop_prob=cfg_dict.get("segmentation_module_dropout", 0.0),
                bias=cfg_dict.get("segmentation_module_bias", False),
            )
        else:
            raise ValueError(f"Segmentation module {segmentation_module} not implemented.")

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
        if self.consecutive_slices > 1 and self.reconstruction_module_dimensionality == 2:
            # Do per slice reconstruction
            pred_reconstruction_slices = []
            for slice_idx in range(self.consecutive_slices):
                y_slice = y[:, slice_idx, ...]
                prediction_slice = y_slice.clone()
                sensitivity_maps_slice = sensitivity_maps[:, slice_idx, ...]
                mask_slice = mask[:, 0, ...]
                init_reconstruction_pred_slice = init_reconstruction_pred[:, slice_idx, ...]
                _pred_reconstruction_slice = (
                    None
                    if init_reconstruction_pred_slice is None or init_reconstruction_pred_slice.dim() < 4
                    else init_reconstruction_pred_slice
                )
                target_reconstruction_slice = target_reconstruction[:, slice_idx, ...]
                hx = None
                sigma = 1.0
                cascades_etas = []
                for i, cascade in enumerate(self.reconstruction_module):
                    # Forward pass through the cascades
                    prediction_slice, hx = cascade(
                        prediction_slice,
                        y_slice,
                        sensitivity_maps_slice,
                        mask_slice,
                        _pred_reconstruction_slice,
                        hx,
                        sigma,
                        keep_eta=False if i == 0 else self.keep_eta,
                    )
                    time_steps_etas = [
                        self.process_intermediate_pred(pred, sensitivity_maps_slice, target_reconstruction_slice)
                        for pred in prediction_slice
                    ]
                    cascades_etas.append(torch.stack(time_steps_etas, dim=0))
                pred_reconstruction_slices.append(torch.stack(cascades_etas, dim=0))
            preds = torch.stack(pred_reconstruction_slices, dim=3)

            cascades_etas = [
                [preds[cascade_eta, time_step_eta, ...] for time_step_eta in range(preds.shape[1])]
                for cascade_eta in range(preds.shape[0])
            ]
        else:
            prediction = y.clone()
            _pred_reconstruction = (
                None
                if init_reconstruction_pred is None or init_reconstruction_pred.dim() < 4
                else init_reconstruction_pred
            )
            hx = None
            sigma = 1.0
            cascades_etas = []
            for i, cascade in enumerate(self.reconstruction_module):
                # Forward pass through the cascades
                prediction, hx = cascade(
                    prediction,
                    y,
                    sensitivity_maps,
                    mask,
                    _pred_reconstruction,
                    hx,
                    sigma,
                    keep_eta=False if i == 0 else self.keep_eta,
                )
                time_steps_etas = [
                    self.process_intermediate_pred(pred, sensitivity_maps, target_reconstruction)
                    for pred in prediction
                ]
                cascades_etas.append(time_steps_etas)
        pred_reconstruction = cascades_etas

        _pred_reconstruction = pred_reconstruction
        if isinstance(_pred_reconstruction, list):
            _pred_reconstruction = _pred_reconstruction[-1]
        if isinstance(_pred_reconstruction, list):
            _pred_reconstruction = _pred_reconstruction[-1]
        if _pred_reconstruction.shape[-1] != 2:  # type: ignore
            _pred_reconstruction = torch.view_as_real(_pred_reconstruction)
        if self.consecutive_slices > 1 and _pred_reconstruction.dim() == 5:
            _pred_reconstruction = _pred_reconstruction.reshape(  # type: ignore
                _pred_reconstruction.shape[0] * _pred_reconstruction.shape[1],  # type: ignore
                *_pred_reconstruction.shape[2:],  # type: ignore
            )
        if _pred_reconstruction.shape[-1] == 2:  # type: ignore
            if self.input_channels == 1:
                _pred_reconstruction = torch.view_as_complex(_pred_reconstruction).unsqueeze(1)
                if self.magnitude_input:
                    _pred_reconstruction = torch.abs(_pred_reconstruction)
            elif self.input_channels == 2:
                if self.magnitude_input:
                    raise ValueError("Magnitude input is not supported for 2-channel input.")
                _pred_reconstruction = _pred_reconstruction.permute(0, 3, 1, 2)  # type: ignore
            else:
                raise ValueError("The input channels must be either 1 or 2. Found: {}".format(self.input_channels))
        else:
            _pred_reconstruction = _pred_reconstruction.unsqueeze(1)

        pred_segmentation = self.segmentation_module(_pred_reconstruction)

        pred_segmentation = torch.abs(pred_segmentation / torch.abs(torch.max(pred_segmentation)))
        if self.consecutive_slices > 1:
            # get batch size and number of slices from y, because if the reconstruction module is used they will not
            # be saved before
            pred_segmentation = pred_segmentation.view([y.shape[0], y.shape[1], *pred_segmentation.shape[1:]])

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
        if not self.no_dc or do_coil_combination:
            pred = ifft2(
                pred, centered=self.fft_centered, normalization=self.fft_normalization, spatial_dims=self.spatial_dims
            )
            pred = coil_combination(pred, sensitivity_maps, method=self.coil_combination_method, dim=self.coil_dim)
        pred = torch.view_as_complex(pred)
        if target.shape[-1] == 2:
            target = torch.view_as_complex(target)
        _, pred = center_crop_to_smallest(target, pred)
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
                    x * torch.logspace(-1, 0, steps=self.reconstruction_module_time_steps).to(time_steps_loss[0])
                    for x in time_steps_loss
                ]
                cascades_loss.append(sum(sum(_loss) / self.reconstruction_module_time_steps))
            yield sum(list(cascades_loss)) / len(self.reconstruction_module)
        else:
            return loss_fn(target, pred)
