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
from mridc.collections.common.parts.fft import fft2c, ifft2c
from mridc.collections.common.parts.rnn_utils import rnn_weights_init
from mridc.collections.common.parts.utils import coil_combination, complex_conj, complex_mul
from mridc.collections.reconstruction.models.base import BaseMRIReconstructionModel, BaseSensitivityModel
from mridc.collections.reconstruction.models.recurrentvarnet.recurentvarnet import RecurrentInit, RecurrentVarNetBlock
from mridc.collections.reconstruction.parts.utils import center_crop_to_smallest
from mridc.core.classes.common import typecheck

__all__ = ["RecurrentVarNet"]


class RecurrentVarNet(BaseMRIReconstructionModel, ABC):
    """
    Recurrent Variational Network implementation as presented in [1]_.

    References
    ----------
    .. [1] Yiasemis, George, et al. “Recurrent Variational Network: A Deep Learning Inverse Problem Solver Applied to t
    he Task of Accelerated MRI Reconstruction.” ArXiv:2111.09639 [Physics], Nov. 2021. arXiv.org,
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

        self.fft_type = cfg_dict.get("fft_type")
        self.output_type = cfg_dict.get("output_type")

        self.block_list: torch.nn.Module = torch.nn.ModuleList()
        for _ in range(self.num_steps if self.no_parameter_sharing else 1):
            self.block_list.append(
                RecurrentVarNetBlock(
                    in_channels=self.in_channels,
                    hidden_channels=self.recurrent_hidden_channels,
                    num_layers=self.recurrent_num_layers,
                    fft_type=self.fft_type,
                )
            )

        # Initialize the sensitivity network if use_sens_net is True
        self.use_sens_net = cfg_dict.get("use_sens_net")
        if self.use_sens_net:
            self.sens_net = BaseSensitivityModel(
                cfg_dict.get("sens_chans"),
                cfg_dict.get("sens_pools"),
                fft_type=self.fft_type,
                mask_type=cfg_dict.get("sens_mask_type"),
                normalize=cfg_dict.get("sens_normalize"),
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
        Args:
            y: torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2], masked kspace data
            sensitivity_maps: torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2], coil sensitivity maps
            mask: torch.Tensor, shape [1, 1, n_x, n_y, 1], sampling mask
            init_pred: torch.Tensor, shape [batch_size, n_x, n_y, 2], initial guess for pred
            target: torch.Tensor, shape [batch_size, n_x, n_y, 2], target data
        Returns:
             Final prediction of the network.
        """
        sensitivity_maps = self.sens_net(y, mask) if self.use_sens_net else sensitivity_maps

        previous_state: Optional[torch.Tensor] = None

        if self.initializer is not None:
            if self.initializer_initialization == "sense":
                initializer_input_image = (
                    complex_mul(ifft2c(y, fft_type=self.fft_type), complex_conj(sensitivity_maps)).sum(1).unsqueeze(1)
                )
            elif self.initializer_initialization == "input_image":
                if "initial_image" not in kwargs:
                    raise ValueError(
                        "`'initial_image` is required as input if initializer_initialization "
                        f"is {self.initializer_initialization}."
                    )
                initializer_input_image = kwargs["initial_image"].unsqueeze(1)
            elif self.initializer_initialization == "zero_filled":
                initializer_input_image = ifft2c(y, fft_type=self.fft_type)

            previous_state = self.initializer(
                fft2c(initializer_input_image, fft_type=self.fft_type).sum(1).permute(0, 3, 1, 2)
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

        eta = ifft2c(kspace_prediction, fft_type=self.fft_type)
        eta = coil_combination(eta, sensitivity_maps, method=self.output_type, dim=1)
        eta = torch.view_as_complex(eta)
        _, eta = center_crop_to_smallest(target, eta)
        return eta
