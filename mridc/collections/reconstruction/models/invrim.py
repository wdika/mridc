# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from abc import ABC
from typing import Generator, Union

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch.nn import L1Loss

from mridc.collections.common.losses.ssim import SSIMLoss
from mridc.collections.common.parts.utils import coil_combination
from mridc.collections.reconstruction.models.base import BaseMRIReconstructionModel, BaseSensitivityModel
from mridc.collections.reconstruction.models.invrim_base.invert_to_learn import MemoryFreeInvertibleModule
from mridc.collections.reconstruction.models.invrim_base.invertible_unet import InvertibleUnet
from mridc.collections.reconstruction.models.invrim_base.irim import IRIM
from mridc.collections.reconstruction.models.invrim_base.residual_blocks import ResidualBlockPixelshuffle
from mridc.collections.reconstruction.models.invrim_base.utils import mse_gradient
from mridc.collections.reconstruction.parts.utils import center_crop_to_smallest
from mridc.core.classes.common import typecheck

__all__ = ["InvRIM"]


class InvRIM(BaseMRIReconstructionModel, ABC):
    """
    Invertible Recurrent Inference Machines (iRIM), as proposed in [1].

    References
    ----------
    .. [1]
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # init superclass
        super().__init__(cfg=cfg, trainer=trainer)

        # Cascades of RIM blocks
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.fft_centered = cfg_dict.get("fft_centered")
        self.fft_normalization = cfg_dict.get("fft_normalization")
        self.spatial_dims = cfg_dict.get("spatial_dims")
        self.coil_dim = cfg_dict.get("coil_dim")

        self.im_channels = cfg_dict.get("im_channels")
        self.multiplicity = cfg_dict.get("multiplicity")
        self.n_hidden = cfg_dict.get("n_hidden")
        self.depth = cfg_dict.get("depth")
        self.dilations = cfg_dict.get("dilations")
        self.n_network_hidden = cfg_dict.get("n_network_hidden")
        self.n_slices = cfg_dict.get("n_slices")
        self.shared_weights = cfg_dict.get("shared_weights")
        self.n_steps = cfg_dict.get("n_steps")

        im_channels = self.im_channels * self.multiplicity
        channels = self.n_hidden if isinstance(self.n_hidden, list) else [self.n_hidden] * self.depth
        self.n_latent = channels[0]
        dilations = self.dilations
        n_hidden = self.n_network_hidden
        conv_nd = 3 if self.n_slices > 1 else 2

        if conv_nd == 3:
            # Make sure to not downsample in the slice direction
            dilations = [[1, d, d] for d in dilations]

        if self.shared_weights:
            cell = torch.nn.ModuleList(
                [InvertibleUnet(n_channels=channels, n_hidden=n_hidden, dilations=dilations, conv_nd=conv_nd)]
                * self.n_steps
            )
        else:
            cell = torch.nn.ModuleList(
                [
                    InvertibleUnet(n_channels=channels, n_hidden=n_hidden, dilations=dilations, conv_nd=conv_nd)
                    for _ in range(self.n_steps)
                ]
            )

        if cfg_dict.get("parametric_output"):
            self.output_function = ResidualBlockPixelshuffle(
                channels[0], 2, channels[0], conv_nd=conv_nd, use_glu=False
            )
        else:

            def real_to_complex(x):
                return torch.stack(torch.chunk(x, 2, 1), -1)

            def complex_to_real(x):
                return torch.cat((x[..., 0], x[..., 1]), 1)

            self.output_function = lambda x: complex_to_real(  # type: ignore
                real_to_complex(x)[:, : im_channels // (2 * self.multiplicity)]
            )

        self.model = MemoryFreeInvertibleModule(
            IRIM(cell, grad_fun=mse_gradient, fft_type=self.fft_type, n_channels=im_channels)
        )

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

        self.train_loss_fn = SSIMLoss() if cfg_dict.get("train_loss_fn") == "ssim" else L1Loss()
        self.eval_loss_fn = SSIMLoss() if cfg_dict.get("eval_loss_fn") == "ssim" else L1Loss()

        self.output_type = cfg_dict.get("output_type")
        self.accumulate_estimates = False

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
        # sensitivity_maps = self.sens_net(y, mask) if self.use_sens_net else sensitivity_maps
        y = torch.cat(self.multiplicity * [y], 1)
        eta = torch.cat([y[..., 0], y[..., 1]], 1)
        x = torch.cat((eta, eta.new_zeros((eta.size(0), self.n_latent - eta.size(1)) + eta.size()[2:])), 1)
        x = self.model.forward(x, [y, mask])
        eta = self.output_function(x)
        eta = torch.stack(torch.chunk(eta, 2, 1), -1)
        eta = torch.view_as_real(eta[..., 0] + 1j * eta[..., 1])
        eta = coil_combination(eta, sensitivity_maps, method=self.output_type, dim=1)
        eta = torch.view_as_complex(eta)
        _, eta = center_crop_to_smallest(target, eta)
        return eta
