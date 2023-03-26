# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from abc import ABC
from typing import List, Tuple, Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

import mridc.collections.multitask.rs.nn.base as base_rs_models
import mridc.core.classes.common as common_classes
from mridc.collections.common.parts import fft, utils
from mridc.collections.multitask.rs.nn.mtlrs_base.mtlrs_block import MTLRSBlock

__all__ = ["MTLRS"]


class MTLRS(base_rs_models.BaseMRIReconstructionSegmentationModel, ABC):  # type: ignore
    """
    Implementation of the Multi-Task Learning for MRI Reconstruction and Segmentation (MTLRS) model, as presented in
    [1].

    References
    ----------
    .. [1] Placeholder.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # init superclass
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.reconstruction_module_recurrent_filters = cfg_dict.get("reconstruction_module_recurrent_filters")
        self.reconstruction_module_time_steps = cfg_dict.get("reconstruction_module_time_steps")
        self.reconstruction_module_num_cascades = cfg_dict.get("reconstruction_module_num_cascades")
        self.reconstruction_module_accumulate_predictions = cfg_dict.get(
            "reconstruction_module_accumulate_predictions"
        )
        conv_dim = cfg_dict.get("reconstruction_module_conv_dim")
        reconstruction_module_params = {
            "num_cascades": self.reconstruction_module_num_cascades,
            "time_steps": self.reconstruction_module_time_steps,
            "no_dc": cfg_dict.get("reconstruction_module_no_dc"),
            "keep_prediction": cfg_dict.get("reconstruction_module_keep_prediction"),
            "dimensionality": cfg_dict.get("reconstruction_module_dimensionality"),
            "recurrent_layer": cfg_dict.get("reconstruction_module_recurrent_layer"),
            "conv_filters": cfg_dict.get("reconstruction_module_conv_filters"),
            "conv_kernels": cfg_dict.get("reconstruction_module_conv_kernels"),
            "conv_dilations": cfg_dict.get("reconstruction_module_conv_dilations"),
            "conv_bias": cfg_dict.get("reconstruction_module_conv_bias"),
            "recurrent_filters": self.reconstruction_module_recurrent_filters,
            "recurrent_kernels": cfg_dict.get("reconstruction_module_recurrent_kernels"),
            "recurrent_dilations": cfg_dict.get("reconstruction_module_recurrent_dilations"),
            "recurrent_bias": cfg_dict.get("reconstruction_module_recurrent_bias"),
            "depth": cfg_dict.get("reconstruction_module_depth"),
            "conv_dim": conv_dim,
            "pretrained": cfg_dict.get("pretrained"),
            "accumulate_predictions": self.reconstruction_module_accumulate_predictions,
        }

        self.segmentation_module_output_channels = cfg_dict.get("segmentation_module_output_channels", 2)
        segmentation_module_params = {
            "segmentation_module": cfg_dict.get("segmentation_module"),
            "output_channels": self.segmentation_module_output_channels,
            "channels": cfg_dict.get("segmentation_module_channels", 64),
            "pooling_layers": cfg_dict.get("segmentation_module_pooling_layers", 2),
            "dropout": cfg_dict.get("segmentation_module_dropout", 0.0),
            "temporal_kernel": cfg_dict.get("segmentation_module_temporal_kernel", 1),
            "activation": cfg_dict.get("segmentation_module_activation", "elu"),
            "bias": cfg_dict.get("segmentation_module_bias", False),
            "conv_dim": conv_dim,
        }

        self.coil_dim = cfg_dict.get("coil_dim", 1)
        self.consecutive_slices = cfg_dict.get("consecutive_slices", 1)

        self.rs_cascades = cfg_dict.get("joint_reconstruction_segmentation_module_cascades", 1)
        self.rs_module = torch.nn.ModuleList(
            [
                MTLRSBlock(
                    reconstruction_module_params=reconstruction_module_params,
                    segmentation_module_params=segmentation_module_params,
                    input_channels=cfg_dict.get("segmentation_module_input_channels", 2),
                    magnitude_input=cfg_dict.get("magnitude_input", False),
                    fft_centered=cfg_dict.get("fft_centered", False),
                    fft_normalization=cfg_dict.get("fft_normalization", "backward"),
                    spatial_dims=cfg_dict.get("spatial_dims", (-2, -1)),
                    coil_dim=self.coil_dim,
                    dimensionality=cfg_dict.get("dimensionality", 2),
                    consecutive_slices=self.consecutive_slices,
                    coil_combination_method=cfg_dict.get("coil_combination_method", "SENSE"),
                    normalize_segmentation_output=cfg_dict.get("normalize_segmentation_output", True),
                )
                for _ in range(self.rs_cascades)
            ]
        )

        self.task_adaption_type = cfg_dict.get("task_adaption_type", "multi_task_learning")

    @common_classes.typecheck()  # type: ignore
    def forward(  # noqa: W0221
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
        pred_reconstructions = []
        for cascade in self.rs_module:
            pred_reconstruction, pred_segmentation, hx = cascade(
                y=y,
                sensitivity_maps=sensitivity_maps,
                mask=mask,
                init_reconstruction_pred=init_reconstruction_pred,
                target_reconstruction=target_reconstruction,
                hx=hx,
                sigma=sigma,
            )
            pred_reconstructions.append(pred_reconstruction)
            init_reconstruction_pred = pred_reconstruction[-1][-1]

            if self.task_adaption_type == "multi_task_learning":
                hidden_states = [
                    torch.cat(
                        [torch.abs(init_reconstruction_pred.unsqueeze(self.coil_dim) * pred_segmentation)]
                        * (f // self.segmentation_module_output_channels),
                        dim=self.coil_dim,
                    )
                    for f in self.reconstruction_module_recurrent_filters
                    if f != 0
                ]

                if self.consecutive_slices > 1:
                    hx = [x.unsqueeze(1) for x in hx]

                # Check if the concatenated hidden states are the same size as the hidden state of the RNN
                if hidden_states[0].shape[self.coil_dim] != hx[0].shape[self.coil_dim]:
                    hidden_states = [
                        torch.cat(
                            [hs, torch.zeros_like(hx[0][:, 0, :, :]).unsqueeze(self.coil_dim)], dim=self.coil_dim
                        )
                        for hs in hidden_states
                        for _ in range(hx[0].shape[1] - hidden_states[0].shape[1])
                    ]

                hx = [hx[i] + hidden_states[i] for i in range(len(hx))]

            init_reconstruction_pred = torch.view_as_real(init_reconstruction_pred)

        return pred_reconstructions, pred_segmentation

    def process_reconstruction_loss(  # noqa: W0221
        self,
        target: torch.Tensor,
        prediction: Union[list, torch.Tensor],
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
        loss_func: torch.nn.Module,
    ) -> torch.Tensor:
        """
        Processes the reconstruction loss.

        Parameters
        ----------
        target : torch.Tensor
            Target data of shape [batch_size, n_x, n_y, 2].
        prediction : Union[list, torch.Tensor]
            Prediction(s) of shape [batch_size, n_x, n_y, 2].
        sensitivity_maps : torch.Tensor
            Sensitivity maps of shape [batch_size, n_coils, n_x, n_y, 2]. It will be used if self.ssdu is True, to
            expand the target and prediction to multiple coils.
        mask : torch.Tensor
            Mask of shape [batch_size, n_x, n_y, 2]. It will be used if self.ssdu is True, to enforce data consistency
            on the prediction.
        loss_func : torch.nn.Module
            Loss function. Must be one of {torch.nn.L1Loss(), torch.nn.MSELoss(),
            mridc.collections.reconstruction.losses.ssim.SSIMLoss()}. Default is ``torch.nn.L1Loss()``.

        Returns
        -------
        loss: torch.FloatTensor
            If self.accumulate_loss is True, returns an accumulative result of all intermediate losses.
            Otherwise, returns the loss of the last intermediate loss.
        """
        if isinstance(target, list):
            target = target[-1]
        if isinstance(target, list):
            target = target[-1]

        if not self.kspace_reconstruction_loss:
            target = torch.abs(target / torch.max(torch.abs(target)))
        else:
            if target.shape[-1] != 2:
                target = torch.view_as_real(target)
            if self.ssdu:
                target = utils.expand_op(target, sensitivity_maps, self.coil_dim)
            target = fft.fft2(target, self.fft_centered, self.fft_normalization, self.spatial_dims)

        if "ssim" in str(loss_func).lower():
            max_value = np.array(torch.max(torch.abs(target)).item()).astype(np.float32)

            def compute_reconstruction_loss(x, y):
                """
                Wrapper for SSIM loss.

                Parameters
                ----------
                x : torch.Tensor
                    Target of shape [batch_size, n_x, n_y, 2].
                y : torch.Tensor
                    Prediction of shape [batch_size, n_x, n_y, 2].

                Returns
                -------
                loss: torch.FloatTensor
                    Loss value.
                """
                y = torch.abs(y / torch.max(torch.abs(y)))
                return loss_func(
                    x.unsqueeze(dim=self.coil_dim),
                    y.unsqueeze(dim=self.coil_dim),
                    data_range=torch.tensor(max_value).unsqueeze(dim=0).to(x.device),
                )

        else:

            def compute_reconstruction_loss(x, y):
                """
                Wrapper for any (expect the SSIM) loss.

                Parameters
                ----------
                x : torch.Tensor
                    Target of shape [batch_size, n_x, n_y, 2].
                y : torch.Tensor
                    Prediction of shape [batch_size, n_x, n_y, 2].

                Returns
                -------
                loss: torch.FloatTensor
                    Loss value.
                """
                if not self.kspace_reconstruction_loss:
                    y = torch.abs(y / torch.max(torch.abs(y)))
                else:
                    if y.shape[-1] != 2:
                        y = torch.view_as_real(y)
                    if self.ssdu:
                        y = utils.expand_op(y, sensitivity_maps, self.coil_dim)
                    y = fft.fft2(y, self.fft_centered, self.fft_normalization, self.spatial_dims)
                    if self.ssdu:
                        y = y * mask
                return loss_func(x, y)

        if self.reconstruction_module_accumulate_predictions:
            rs_cascades_loss = []
            for rs_cascade in prediction:
                cascades_loss = []
                for cascade_pred in rs_cascade:
                    time_steps_loss = [
                        compute_reconstruction_loss(target, time_step_pred) for time_step_pred in cascade_pred
                    ]
                    _loss = [
                        x * torch.logspace(-1, 0, steps=self.reconstruction_module_time_steps).to(time_steps_loss[0])
                        for x in time_steps_loss
                    ]
                    cascades_loss.append(sum(sum(_loss) / self.reconstruction_module_time_steps))
                rs_cascades_loss.append(sum(list(cascades_loss)) / self.reconstruction_module_num_cascades)
            return sum(list(rs_cascades_loss)) / self.rs_cascades
        return compute_reconstruction_loss(target, prediction) * self.loss_regularization_factor
