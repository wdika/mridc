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
from mridc.collections.common.parts.fft import fft2, ifft2
from mridc.collections.common.parts.rnn_utils import rnn_weights_init
from mridc.collections.common.parts.utils import coil_combination
from mridc.collections.reconstruction.models.base import BaseMRIReconstructionModel, BaseSensitivityModel
from mridc.collections.reconstruction.models.cs.cg import conjugate_gradient
from mridc.collections.reconstruction.models.cs.pocs import POCS
from mridc.collections.reconstruction.models.rim.rim_block import RIMBlock
from mridc.collections.reconstruction.parts.utils import center_crop_to_smallest
from mridc.core.classes.common import typecheck

__all__ = ["SRCIRIM"]


class SRCIRIM(BaseMRIReconstructionModel, ABC):
    """SR Cascades of Independently Recurrent Inference Machines implementation."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # init superclass
        super().__init__(cfg=cfg, trainer=trainer)

        # Cascades of RIM blocks
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.fft_centered = cfg_dict.get("fft_centered")
        self.fft_normalization = cfg_dict.get("fft_normalization")
        self.spatial_dims = cfg_dict.get("spatial_dims")
        self.coil_dim = cfg_dict.get("coil_dim")
        self.no_dc = cfg_dict.get("no_dc")

        self.peaks_repetitions = cfg_dict.get("peaks_repetitions")
        self.upsampling_cascades = cfg_dict.get("upsampling_cascades")
        self.channels = cfg_dict.get("channels")

        # make time-steps size divisible by 2 for fast fp16 training
        self.time_steps = 2 * math.ceil(cfg_dict.get("time_steps") / 2)

        self.net = torch.nn.ModuleList([])
        # Create signal repetitions
        for _ in range(self.peaks_repetitions):
            for num_cascades in self.upsampling_cascades:
                time_steps = self.time_steps
                channels = self.channels
                # Create upsampling cascades
                for _ in range(num_cascades):
                    self.net.append(
                        RIMBlock(
                            recurrent_layer=cfg_dict.get("recurrent_layer"),
                            conv_filters=[channels, channels, 2],
                            conv_kernels=cfg_dict.get("conv_kernels"),
                            conv_dilations=cfg_dict.get("conv_dilations"),
                            conv_bias=cfg_dict.get("conv_bias"),
                            recurrent_filters=[channels, channels, 0],
                            recurrent_kernels=cfg_dict.get("recurrent_kernels"),
                            recurrent_dilations=cfg_dict.get("recurrent_dilations"),
                            recurrent_bias=cfg_dict.get("recurrent_bias"),
                            depth=cfg_dict.get("depth"),
                            time_steps=time_steps,
                            conv_dim=cfg_dict.get("conv_dim"),
                            no_dc=self.no_dc,
                            fft_centered=self.fft_centered,
                            fft_normalization=self.fft_normalization,
                            spatial_dims=self.spatial_dims,
                            coil_dim=self.coil_dim,
                            dimensionality=cfg_dict.get("dimensionality"),
                        )
                    )
                    channels *= 2
                    time_steps = time_steps + 2
                channels //= 2
                time_steps = time_steps - 2
                # Create downsampling cascades
                for _ in range(num_cascades - 1):
                    channels //= 2
                    time_steps = time_steps - 2
                    self.net.append(
                        RIMBlock(
                            recurrent_layer=cfg_dict.get("recurrent_layer"),
                            conv_filters=[channels, channels, 2],
                            conv_kernels=cfg_dict.get("conv_kernels"),
                            conv_dilations=cfg_dict.get("conv_dilations"),
                            conv_bias=cfg_dict.get("conv_bias"),
                            recurrent_filters=[channels, channels, 0],
                            recurrent_kernels=cfg_dict.get("recurrent_kernels"),
                            recurrent_dilations=cfg_dict.get("recurrent_dilations"),
                            recurrent_bias=cfg_dict.get("recurrent_bias"),
                            depth=cfg_dict.get("depth"),
                            time_steps=time_steps,
                            conv_dim=cfg_dict.get("conv_dim"),
                            no_dc=self.no_dc,
                            fft_centered=self.fft_centered,
                            fft_normalization=self.fft_normalization,
                            spatial_dims=self.spatial_dims,
                            coil_dim=self.coil_dim,
                            dimensionality=cfg_dict.get("dimensionality"),
                        )
                    )

            # for num_cascades in self.upsampling_cascades[::-1][1:]:
            #     time_steps = self.time_steps
            #     channels = self.channels
            #     # Create upsampling cascades
            #     for _ in range(num_cascades):
            #         self.net.append(
            #             RIMBlock(
            #                 recurrent_layer=cfg_dict.get("recurrent_layer"),
            #                 conv_filters=[channels, channels, 2],
            #                 conv_kernels=cfg_dict.get("conv_kernels"),
            #                 conv_dilations=cfg_dict.get("conv_dilations"),
            #                 conv_bias=cfg_dict.get("conv_bias"),
            #                 recurrent_filters=[channels, channels, 0],
            #                 recurrent_kernels=cfg_dict.get("recurrent_kernels"),
            #                 recurrent_dilations=cfg_dict.get("recurrent_dilations"),
            #                 recurrent_bias=cfg_dict.get("recurrent_bias"),
            #                 depth=cfg_dict.get("depth"),
            #                 time_steps=time_steps,
            #                 conv_dim=cfg_dict.get("conv_dim"),
            #                 no_dc=self.no_dc,
            #                 body_coil_regularization=cfg_dict.get("body_coil_regularization"),
            #                 body_coil_regularization_epsilon=cfg_dict.get("body_coil_regularization_epsilon"),
            #                 body_coil_regularization_gamma=cfg_dict.get("body_coil_regularization_gamma"),
            #                 fft_type=self.fft_type,
            #             )
            #         )
            #         channels *= 2
            #         time_steps = time_steps + 2
            #     channels //= 2
            #     time_steps = time_steps - 2
            #     # Create downsampling cascades
            #     for _ in range(num_cascades - 1):
            #         channels //= 2
            #         time_steps = time_steps - 2
            #         self.net.append(
            #             RIMBlock(
            #                 recurrent_layer=cfg_dict.get("recurrent_layer"),
            #                 conv_filters=[channels, channels, 2],
            #                 conv_kernels=cfg_dict.get("conv_kernels"),
            #                 conv_dilations=cfg_dict.get("conv_dilations"),
            #                 conv_bias=cfg_dict.get("conv_bias"),
            #                 recurrent_filters=[channels, channels, 0],
            #                 recurrent_kernels=cfg_dict.get("recurrent_kernels"),
            #                 recurrent_dilations=cfg_dict.get("recurrent_dilations"),
            #                 recurrent_bias=cfg_dict.get("recurrent_bias"),
            #                 depth=cfg_dict.get("depth"),
            #                 time_steps=time_steps,
            #                 conv_dim=cfg_dict.get("conv_dim"),
            #                 no_dc=self.no_dc,
            #                 body_coil_regularization=cfg_dict.get("body_coil_regularization"),
            #                 body_coil_regularization_epsilon=cfg_dict.get("body_coil_regularization_epsilon"),
            #                 body_coil_regularization_gamma=cfg_dict.get("body_coil_regularization_gamma"),
            #                 fft_type=self.fft_type,
            #             )
            #         )

        # Keep estimation through the cascades if keep_eta is True or re-estimate it if False.
        self.keep_eta = cfg_dict.get("keep_eta")
        self.output_type = cfg_dict.get("output_type")

        # Initialize the sensitivity network if use_sens_net is True
        self.use_sens_net = cfg_dict.get("use_sens_net")
        if self.use_sens_net:
            self.sens_net = BaseSensitivityModel(
                cfg_dict.get("sens_chans"),
                cfg_dict.get("sens_pools"),
                fft_centered=self.fft_centered,
                fft_normalization=self.fft_normalization,
                spatial_dims=self.spatial_dims,
                coil_dim=self.coil_dim,
                mask_type=cfg_dict.get("sens_mask_type"),
                normalize=cfg_dict.get("sens_normalize"),
                mask_center=cfg_dict.get("sens_mask_center"),
            )

        # initialize weights if not using pretrained cirim
        if not cfg_dict.get("pretrained", False):
            std_init_range = 1 / channels**0.5
            self.net.apply(lambda module: rnn_weights_init(module, std_init_range))

        self.train_loss_fn = SSIMLoss() if cfg_dict.get("train_loss_fn") == "ssim" else L1Loss()
        self.eval_loss_fn = SSIMLoss() if cfg_dict.get("eval_loss_fn") == "ssim" else L1Loss()

        self.dc_weight = torch.nn.Parameter(torch.ones(1))
        self.accumulate_estimates = True

        self.use_pocs = cfg_dict.get("use_pocs")
        if self.use_pocs:
            self.pocs_soft_threshold_val = cfg_dict.get("pocs_soft_threshold_val")
            self.pocs_max_iter = cfg_dict.get("pocs_max_iter")

        self.use_cg = cfg_dict.get("use_cg")
        if self.use_cg:
            self.cg_threshold_val = cfg_dict.get("cg_threshold_val")
            self.cg_max_iter = cfg_dict.get("cg_max_iter")
            self.cg_approx_param = cfg_dict.get("cg_approx_param")
            self.cg_alpha = cfg_dict.get("cg_alpha")
            self.cg_beta = cfg_dict.get("cg_beta")

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
        if self.use_pocs:
            y = fft2(
                POCS(
                    y,
                    mask,
                    self.fft_centered,
                    self.fft_normalization,
                    self.spatial_dims,
                    self.pocs_soft_threshold_val,
                    self.pocs_max_iter,
                ),
                self.fft_centered,
                self.fft_normalization,
                self.spatial_dims,
            )

        if self.use_cg:
            y = fft2(
                conjugate_gradient(
                    y,
                    mask,
                    self.cg_threshold_val,
                    self.cg_max_iter,
                    self.cg_approx_param,
                    self.cg_alpha,
                    self.cg_beta,
                    self.fft_centered,
                    self.fft_normalization,
                    self.spatial_dims,
                )[-1],
                self.fft_centered,
                self.fft_normalization,
                self.spatial_dims,
            )

        sensitivity_maps = self.sens_net(y, mask) if self.use_sens_net else sensitivity_maps
        prediction = y.clone()
        init_pred = None if init_pred is None or init_pred.dim() < 4 else init_pred
        hx = None
        sigma = 1.0
        cascades_etas = []
        for i, cascade in enumerate(self.net):
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
            pred = ifft2(
                pred,
                self.fft_centered,
                self.fft_normalization,
                self.spatial_dims,
            )
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
            # Calculate weights for the loss for each time step
            ts_weights = []
            c_weights = []
            for _ in range(self.peaks_repetitions):
                for num_cascades in self.upsampling_cascades:
                    # Calculate weights for the loss for each upsampling cascade
                    c_weights.append(torch.logspace(-1, 0, steps=num_cascades).to(target))
                    # Calculate weights for the loss for each time step
                    time_steps = self.time_steps
                    for i in range(num_cascades):
                        ts_weights.append(torch.logspace(-1, 0, steps=time_steps).to(target))
                        time_steps = time_steps + 2
                    # Calculate the loss for each downsampling cascade
                    c_weights.append(torch.flip(c_weights[-1], dims=[0])[1:])
                    # Calculate weights for the loss for each time step
                    time_steps = time_steps - 2
                    for i in range(num_cascades - 1):
                        time_steps = time_steps - 2
                        ts_weights.append(torch.logspace(-1, 0, steps=time_steps).to(target))

                # for num_cascades in self.upsampling_cascades[::-1][1:]:
                #     # Calculate weights for the loss for each upsampling cascade
                #     c_weights.append(torch.logspace(-1, 0, steps=num_cascades).to(target))
                #     # Calculate weights for the loss for each time step
                #     time_steps = self.time_steps
                #     for _ in range(num_cascades):
                #         ts_weights.append(torch.logspace(-1, 0, steps=time_steps).to(target))
                #         time_steps = time_steps + 2
                #     # Calculate the loss for each downsampling cascade
                #     c_weights.append(torch.flip(c_weights[-1], dims=[0])[1:])
                #     # Calculate weights for the loss for each time step
                #     time_steps = time_steps - 2
                #     for _ in range(num_cascades - 1):
                #         time_steps = time_steps - 2
                #         ts_weights.append(torch.logspace(-1, 0, steps=time_steps).to(target))

            _c_weights = []
            for cascade_weights in c_weights:
                cascade_weights = cascade_weights.tolist()
                for x in cascade_weights:
                    _c_weights.append(torch.tensor(x).to(target))

            cascades_loss = []
            for j, (cascade_eta, cascade_ts_weights) in enumerate(zip(eta, ts_weights)):
                # Compute the loss for each time step
                _loss = [loss_fn(target, x) for x in cascade_eta]
                # Weight the loss for each time step
                _loss = [_loss[i] * cascade_ts_weights[i].to(_loss[0]) for i in range(len(_loss))]
                # Weight the loss for the current cascade
                # _loss = [_loss[i] * _c_weights[i] for i in range(len(_loss))]
                _loss = [_loss[i] * _c_weights[j] for i in range(len(_loss))]
                # Average the current cascade loss
                cascades_loss.append(sum(_loss) / len(_loss))
            # Average the final loss
            yield sum(cascades_loss) / len(cascades_loss)
        else:
            return loss_fn(target, eta)
