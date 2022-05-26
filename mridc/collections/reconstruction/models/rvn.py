# coding=utf-8
__author__ = "Dimitrios Karkalousos"

import math
from abc import ABC
from typing import Optional

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch.nn import L1Loss

from mridc.collections.common.losses.ssim import SSIMLoss
from mridc.collections.common.parts.fft import fft2, ifft2
from mridc.collections.common.parts.rnn_utils import rnn_weights_init
from mridc.collections.common.parts.utils import coil_combination, complex_conj, complex_mul
from mridc.collections.reconstruction.models.base import BaseMRIReconstructionModel
from mridc.collections.reconstruction.models.recurrentvarnet.recurentvarnet import RecurrentInit, RecurrentVarNetBlock
from mridc.collections.reconstruction.parts.utils import center_crop_to_smallest
from mridc.core.classes.common import typecheck

__all__ = ["RecurrentVarNet"]


class RecurrentVarNet(BaseMRIReconstructionModel, ABC):
    """
    Implementation of the Recurrent Variational Network implementation, as presented in Yiasemis, George, et al.

    References
    ----------

    ..

        Yiasemis, George, et al. “Recurrent Variational Network: A Deep Learning Inverse Problem Solver Applied to \
        the Task of Accelerated MRI Reconstruction.” ArXiv:2111.09639 [Physics], Nov. 2021. arXiv.org, \
        http://arxiv.org/abs/2111.09639.

    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # init superclass
        super().__init__(cfg=cfg, trainer=trainer)

        # Cascades of RIM blocks
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.in_channels = cfg_dict.get("in_channels")
        self.recurrent_hidden_channels = cfg_dict.get("recurrent_hidden_channels")
        self.recurrent_num_layers = cfg_dict.get("recurrent_num_layers")
        self.no_parameter_sharing = cfg_dict.get("no_parameter_sharing")

        # make time-steps size divisible by 8 for fast fp16 training
        self.num_steps = 8 * math.ceil(cfg_dict.get("num_steps") / 8)

        self.learned_initializer = cfg_dict.get("learned_initializer")
        self.initializer_initialization = cfg_dict.get("initializer_initialization")
        self.initializer_channels = cfg_dict.get("initializer_channels")
        self.initializer_dilations = cfg_dict.get("initializer_dilations")

        if (
            self.learned_initializer
            and self.initializer_initialization is not None
            and self.initializer_channels is not None
            and self.initializer_dilations is not None
        ):
            if self.initializer_initialization not in [
                "sense",
                "input_image",
                "zero_filled",
            ]:
                raise ValueError(
                    "Unknown initializer_initialization. Expected `sense`, `'input_image` or `zero_filled`."
                    f"Got {self.initializer_initialization}."
                )
            self.initializer = RecurrentInit(
                self.in_channels,
                self.recurrent_hidden_channels,
                channels=self.initializer_channels,
                dilations=self.initializer_dilations,
                depth=self.recurrent_num_layers,
                multiscale_depth=cfg_dict.get("initializer_multiscale"),
            )
        else:
            self.initializer = None  # type: ignore

        self.fft_centered = cfg_dict.get("fft_centered")
        self.fft_normalization = cfg_dict.get("fft_normalization")
        self.spatial_dims = cfg_dict.get("spatial_dims")
        self.coil_dim = cfg_dict.get("coil_dim")
        self.coil_combination_method = cfg_dict.get("coil_combination_method")

        self.block_list: torch.nn.Module = torch.nn.ModuleList()
        for _ in range(self.num_steps if self.no_parameter_sharing else 1):
            self.block_list.append(
                RecurrentVarNetBlock(
                    in_channels=self.in_channels,
                    hidden_channels=self.recurrent_hidden_channels,
                    num_layers=self.recurrent_num_layers,
                    fft_centered=self.fft_centered,
                    fft_normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                    coil_dim=self.coil_dim,
                )
            )

        std_init_range = 1 / self.recurrent_hidden_channels**0.5

        # initialize weights if not using pretrained cirim
        if not cfg_dict.get("pretrained", False):
            self.block_list.apply(lambda module: rnn_weights_init(module, std_init_range))

        self.train_loss_fn = SSIMLoss() if cfg_dict.get("train_loss_fn") == "ssim" else L1Loss()
        self.eval_loss_fn = SSIMLoss() if cfg_dict.get("eval_loss_fn") == "ssim" else L1Loss()

        self.accumulate_estimates = False

    @typecheck()
    def forward(
        self,
        y: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
        init_pred: torch.Tensor,
        target: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass of the network.

        Parameters
        ----------
        y: Subsampled k-space data.
            torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
        sensitivity_maps: Coil sensitivity maps.
            torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
        mask: Sampling mask.
            torch.Tensor, shape [1, 1, n_x, n_y, 1]
        init_pred: Initial prediction.
            torch.Tensor, shape [batch_size, n_x, n_y, 2]
        target: Target data to compute the loss.
            torch.Tensor, shape [batch_size, n_x, n_y, 2]

        Returns
        -------
        pred: list of torch.Tensor, shape [batch_size, n_x, n_y, 2], or  torch.Tensor, shape [batch_size, n_x, n_y, 2]
             If self.accumulate_loss is True, returns a list of all intermediate estimates.
             If False, returns the final estimate.
        """
        previous_state: Optional[torch.Tensor] = None

        if self.initializer is not None:
            if self.initializer_initialization == "sense":
                initializer_input_image = (
                    complex_mul(
                        ifft2(
                            y,
                            centered=self.fft_centered,
                            normalization=self.fft_normalization,
                            spatial_dims=self.spatial_dims,
                        ),
                        complex_conj(sensitivity_maps),
                    )
                    .sum(self.coil_dim)
                    .unsqueeze(self.coil_dim)
                )
            elif self.initializer_initialization == "input_image":
                if "initial_image" not in kwargs:
                    raise ValueError(
                        "`'initial_image` is required as input if initializer_initialization "
                        f"is {self.initializer_initialization}."
                    )
                initializer_input_image = kwargs["initial_image"].unsqueeze(self.coil_dim)
            elif self.initializer_initialization == "zero_filled":
                initializer_input_image = ifft2(
                    y,
                    centered=self.fft_centered,
                    normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                )

            previous_state = self.initializer(
                fft2(
                    initializer_input_image,
                    centered=self.fft_centered,
                    normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                )
                .sum(1)
                .permute(0, 3, 1, 2)
            )

        kspace_prediction = y.clone()

        for step in range(self.num_steps):
            block = self.block_list[step] if self.no_parameter_sharing else self.block_list[0]
            kspace_prediction, previous_state = block(
                kspace_prediction,
                y,
                mask,
                sensitivity_maps,
                previous_state,
            )

        eta = ifft2(
            kspace_prediction,
            centered=self.fft_centered,
            normalization=self.fft_normalization,
            spatial_dims=self.spatial_dims,
        )
        eta = coil_combination(eta, sensitivity_maps, method=self.coil_combination_method, dim=self.coil_dim)
        eta = torch.view_as_complex(eta)
        _, eta = center_crop_to_smallest(target, eta)
        return eta
