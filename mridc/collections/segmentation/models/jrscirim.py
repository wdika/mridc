# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from abc import ABC
from typing import List, Tuple, Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

import mridc.collections.segmentation.models.base as base_segmentation_models
import mridc.core.classes.common as common_classes

__all__ = ["JRSCIRIM"]

from mridc.collections.segmentation.models.jrscirim_base.jrscirim_block import JRSCIRIMBlock


class JRSCIRIM(base_segmentation_models.BaseMRIJointReconstructionSegmentationModel, ABC):
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

        self.reconstruction_module_recurrent_filters = cfg_dict.get("reconstruction_module_recurrent_filters")
        self.reconstruction_module_time_steps = cfg_dict.get("reconstruction_module_time_steps")
        self.reconstruction_module_num_cascades = cfg_dict.get("reconstruction_module_num_cascades")
        self.reconstruction_module_accumulate_estimates = cfg_dict.get("reconstruction_module_accumulate_estimates")
        conv_dim = cfg_dict.get("reconstruction_module_conv_dim")
        reconstruction_module_params = {
            "num_cascades": self.reconstruction_module_num_cascades,
            "time_steps": self.reconstruction_module_time_steps,
            "no_dc": cfg_dict.get("reconstruction_module_no_dc"),
            "keep_eta": cfg_dict.get("reconstruction_module_keep_eta"),
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
            "accumulate_estimates": self.reconstruction_module_accumulate_estimates,
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

        self.jrs_cascades = cfg_dict.get("joint_reconstruction_segmentation_module_cascades", 1)
        self.jrs_module = torch.nn.ModuleList(
            [
                JRSCIRIMBlock(
                    reconstruction_module_params=reconstruction_module_params,
                    segmentation_module_params=segmentation_module_params,
                    input_channels=cfg_dict.get("segmentation_module_input_channels", 2),
                    magnitude_input=cfg_dict.get("magnitude_input", False),
                    fft_centered=cfg_dict.get("fft_centered", False),
                    fft_normalization=cfg_dict.get("fft_normalization", "ortho"),
                    spatial_dims=cfg_dict.get("spatial_dims", (-2, -1)),
                    coil_dim=self.coil_dim,
                    dimensionality=cfg_dict.get("dimensionality", 2),
                    consecutive_slices=cfg_dict.get("consecutive_slices", 1),
                    coil_combination_method=cfg_dict.get("coil_combination_method", "SENSE"),
                )
                for _ in range(self.jrs_cascades)
            ]
        )

        self.task_adaption_type = cfg_dict.get("task_adaption_type", "joint")
        if self.task_adaption_type not in ("end-to-end", "joint"):
            raise ValueError(
                f"Task adaption type '{self.task_adaption_type}' not supported. "
                "It must be either 'end-to-end' or 'joint'."
            )

    @common_classes.typecheck()
    def forward(
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
        y: Undersampled k-space data.
            torch.Tensor, shape [batch_size, n_echoes, n_coils, n_x, n_y, 2]
        sensitivity_maps: Coil sensitivity maps.
            torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
        mask: Sub-sampling mask.
            torch.Tensor, shape [batch_size, 1, n_x, n_y, 2]
        init_reconstruction_pred: Initial reconstruction prediction.
            torch.Tensor, shape [batch_size, 1, n_x, n_y, 2]
        target_reconstruction: Target reconstruction.
            torch.Tensor, shape [batch_size, 1, n_x, n_y, 2]
        hx: Hidden state of the RNN.
            torch.Tensor, shape [batch_size, n_x, n_y, 2]
        sigma: Noise standard deviation.
            float

        Returns
        -------
        pred_reconstruction: Predicted reconstruction.
            torch.Tensor, shape [batch_size, n_x, n_y, 2]
        pred_segmentation: Predicted segmentation.
            torch.Tensor, shape [batch_size, nr_classes, n_x, n_y]
        """
        pred_reconstructions = []
        for cascade in self.jrs_module:
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

            if self.task_adaption_type == "joint":
                hidden_states = [
                    torch.cat(
                        [torch.abs(init_reconstruction_pred.unsqueeze(self.coil_dim) * pred_segmentation)]
                        * (f // self.segmentation_module_output_channels),
                        dim=self.coil_dim,
                    )
                    for f in self.reconstruction_module_recurrent_filters
                    if f != 0
                ]
                hx = [hx[i] + hidden_states[i] for i in range(len(hx))]

            init_reconstruction_pred = torch.view_as_real(init_reconstruction_pred)

        return pred_reconstructions, pred_segmentation

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
            jrs_cascades_loss = []
            for jrs_cascade in pred:
                cascades_loss = []
                for cascade_pred in jrs_cascade:
                    time_steps_loss = [loss_fn(target, time_step_pred) for time_step_pred in cascade_pred]
                    _loss = [
                        x * torch.logspace(-1, 0, steps=self.reconstruction_module_time_steps).to(time_steps_loss[0])
                        for x in time_steps_loss
                    ]
                    cascades_loss.append(sum(sum(_loss) / self.reconstruction_module_time_steps))
                jrs_cascades_loss.append(sum(list(cascades_loss)) / self.reconstruction_module_num_cascades)
            return sum(list(jrs_cascades_loss)) / self.jrs_cascades
        else:
            return loss_fn(target, pred)
