# coding=utf-8
__author__ = "Dimitrios Karkalousos"

import math
from abc import ABC
from typing import Generator, Union

import numpy as np
import torch

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch.nn import L1Loss

from mridc.collections.common.losses.ssim import SSIMLoss
from mridc.collections.common.parts.fft import ifft2c
from mridc.collections.common.parts.rnn_utils import rnn_weights_init
from mridc.collections.common.parts.utils import coil_combination
from mridc.collections.reconstruction.models.base import BaseMRIReconstructionModel, BaseSensitivityModel
from mridc.collections.reconstruction.models.rim.rim_block import RIMBlock
from mridc.collections.reconstruction.parts.utils import center_crop_to_smallest
from mridc.core.classes.common import typecheck

__all__ = ["CIRIM"]


class CIRIM(BaseMRIReconstructionModel, ABC):
    """
    Cascades of Independently Recurrent Inference Machines implementation as presented in [1]_.

    References
    ----------
    .. [1] Karkalousos, D. et al. (2021) ‘Assessment of Data Consistency through Cascades of Independently Recurrent
    Inference Machines for fast and robust accelerated MRI reconstruction’.
    Available at: https://arxiv.org/abs/2111.15498v1
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # init superclass
        super().__init__(cfg=cfg, trainer=trainer)

        # Cascades of RIM blocks
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.recurrent_filters = cfg_dict.get("recurrent_filters")

        # make time-steps size divisible by 8 for fast fp16 training
        self.time_steps = 8 * math.ceil(cfg_dict.get("time_steps") / 8)

        self.no_dc = cfg_dict.get("no_dc")
        self.fft_type = cfg_dict.get("fft_type")
        self.num_cascades = cfg_dict.get("num_cascades")

        self.cirim = torch.nn.ModuleList(
            [
                RIMBlock(
                    recurrent_layer=cfg_dict.get("recurrent_layer"),
                    conv_filters=cfg_dict.get("conv_filters"),
                    conv_kernels=cfg_dict.get("conv_kernels"),
                    conv_dilations=cfg_dict.get("conv_dilations"),
                    conv_bias=cfg_dict.get("conv_bias"),
                    recurrent_filters=self.recurrent_filters,
                    recurrent_kernels=cfg_dict.get("recurrent_kernels"),
                    recurrent_dilations=cfg_dict.get("recurrent_dilations"),
                    recurrent_bias=cfg_dict.get("recurrent_bias"),
                    depth=cfg_dict.get("depth"),
                    time_steps=self.time_steps,
                    conv_dim=cfg_dict.get("conv_dim"),
                    no_dc=self.no_dc,
                    fft_type=self.fft_type,
                )
                for _ in range(self.num_cascades)
            ]
        )

        # Keep estimation through the cascades if keep_eta is True or re-estimate it if False.
        self.keep_eta = cfg_dict.get("keep_eta")
        self.output_type = cfg_dict.get("output_type")

        # Initialize the sensitivity network if use_sens_net is True
        self.use_sens_net = cfg_dict.get("use_sens_net")
        if self.use_sens_net:
            self.sens_net = BaseSensitivityModel(
                cfg_dict.get("sens_chans"),
                cfg_dict.get("sens_pools"),
                fft_type=self.fft_type,
                mask_type=cfg_dict.get("sens_mask_type"),
                normalize=cfg_dict.get("sens_normalize"),
                mask_center=cfg_dict.get("sens_mask_center"),
            )

        # initialize weights if not using pretrained cirim
        if not cfg_dict.get("pretrained", False):
            std_init_range = 1 / self.recurrent_filters[0] ** 0.5
            self.cirim.apply(lambda module: rnn_weights_init(module, std_init_range))

        self.train_loss_fn = SSIMLoss() if cfg_dict.get("train_loss_fn") == "ssim" else L1Loss()
        self.eval_loss_fn = SSIMLoss() if cfg_dict.get("eval_loss_fn") == "ssim" else L1Loss()

        self.dc_weight = torch.nn.Parameter(torch.ones(1))
        self.accumulate_estimates = True

    @typecheck()
    def forward(
        self,
        y: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
        init_pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Union[Generator, torch.Tensor]:
        """
        Forward pass of the network.
        Args:
            y: torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2], masked kspace data
            sensitivity_maps: torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2], coil sensitivity maps
            mask: torch.Tensor, shape [1, 1, n_x, n_y, 1], sampling mask
            init_pred: torch.Tensor, shape [batch_size, n_x, n_y, 2], initial guess for pred
            target: torch.Tensor, shape [batch_size, n_x, n_y, 2], target data
        Returns:
             Final estimation of the network.
             If self.accumulate_loss is True, returns a list of all intermediate estimates.
             If False, returns the final estimate.
        """
        sensitivity_maps = self.sens_net(y, mask) if self.use_sens_net else sensitivity_maps
        prediction = y.clone()
        init_pred = None if init_pred is None or init_pred.dim() < 4 else init_pred
        hx = None
        sigma = 1.0
        cascades_etas = []
        for i, cascade in enumerate(self.cirim):
            # Forward pass through the cascades
            prediction, hx = cascade(
                prediction,
                y,
                sensitivity_maps,
                mask,
                init_pred,
                hx,
                sigma,
                keep_eta=False if i == 0 else self.keep_eta,
            )
            time_steps_etas = [self.process_intermediate_pred(pred, sensitivity_maps, target) for pred in prediction]
            cascades_etas.append(time_steps_etas)
        yield cascades_etas

    def process_intermediate_pred(self, pred, sensitivity_maps, target, do_coil_combination=False):
        """Process the intermediate eta to be used in the loss function."""
        # Take the last time step of the eta
        if not self.no_dc or do_coil_combination:
            pred = ifft2c(pred, fft_type=self.fft_type)
            pred = coil_combination(pred, sensitivity_maps, method=self.output_type, dim=1)
        pred = torch.view_as_complex(pred)
        _, pred = center_crop_to_smallest(target, pred)
        return pred

    def process_loss(self, target, eta, _loss_fn):
        """Calculate the loss."""
        target = torch.abs(target / torch.max(torch.abs(target)))
        if "ssim" in str(_loss_fn).lower():
            max_value = np.array(torch.max(torch.abs(target)).item()).astype(np.float32)

            def loss_fn(x, y):
                """Calculate the ssim loss."""
                return _loss_fn(
                    x.unsqueeze(dim=1),
                    torch.abs(y / torch.max(torch.abs(y))).unsqueeze(dim=1),
                    data_range=torch.tensor(max_value).unsqueeze(dim=0).to(x.device),
                )

        else:

            def loss_fn(x, y):
                """Calculate other loss."""
                return _loss_fn(x, torch.abs(y / torch.max(torch.abs(y))))

        if self.accumulate_estimates:
            cascades_loss = []
            for cascade_eta in eta:
                time_steps_loss = [loss_fn(target, time_step_eta) for time_step_eta in cascade_eta]
                _loss = [
                    x * torch.logspace(-1, 0, steps=self.time_steps).to(time_steps_loss[0]) for x in time_steps_loss
                ]
                cascades_loss.append(sum(sum(_loss) / self.time_steps))
            yield sum(list(cascades_loss)) / len(self.cirim)
        else:
            return loss_fn(target, eta)
