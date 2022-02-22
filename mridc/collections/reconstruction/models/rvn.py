# coding=utf-8
__author__ = "Dimitrios Karkalousos"

import math
from abc import ABC
from typing import Dict, Generator, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
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
        rvn_cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.in_channels = rvn_cfg_dict.get("in_channels")
        self.recurrent_hidden_channels = rvn_cfg_dict.get("recurrent_hidden_channels")
        self.recurrent_num_layers = rvn_cfg_dict.get("recurrent_num_layers")
        self.no_parameter_sharing = rvn_cfg_dict.get("no_parameter_sharing")

        # make time-steps size divisible by 8 for fast fp16 training
        self.num_steps = 8 * math.ceil(rvn_cfg_dict.get("num_steps") / 8)

        self.learned_initializer = rvn_cfg_dict.get("learned_initializer")
        self.initializer_initialization = rvn_cfg_dict.get("initializer_initialization")
        self.initializer_channels = rvn_cfg_dict.get("initializer_channels")
        self.initializer_dilations = rvn_cfg_dict.get("initializer_dilations")

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
                    f"Unknown initializer_initialization. Expected `sense`, `'input_image` or `zero_filled`."
                    f"Got {self.initializer_initialization}."
                )
            self.initializer = RecurrentInit(
                self.in_channels,
                self.recurrent_hidden_channels,
                channels=self.initializer_channels,
                dilations=self.initializer_dilations,
                depth=self.recurrent_num_layers,
                multiscale_depth=rvn_cfg_dict.get("initializer_multiscale"),
            )
        else:
            self.initializer = None  # type: ignore

        self.fft_type = rvn_cfg_dict.get("fft_type")
        self.output_type = rvn_cfg_dict.get("output_type")

        self.block_list: nn.Module = nn.ModuleList()
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
        self.use_sens_net = rvn_cfg_dict.get("use_sens_net")
        if self.use_sens_net:
            self.sens_net = BaseSensitivityModel(
                rvn_cfg_dict.get("sens_chans"),
                rvn_cfg_dict.get("sens_pools"),
                fft_type=self.fft_type,
                mask_type=rvn_cfg_dict.get("sens_mask_type"),
                normalize=rvn_cfg_dict.get("sens_normalize"),
            )

        std_init_range = 1 / self.recurrent_hidden_channels**0.5

        # initialize weights if not using pretrained cirim
        if not rvn_cfg_dict.get("pretrained", False):
            self.block_list.apply(lambda module: rnn_weights_init(module, std_init_range))

        self.train_loss_fn = SSIMLoss() if rvn_cfg_dict.get("train_loss_fn") == "ssim" else L1Loss()
        self.eval_loss_fn = SSIMLoss() if rvn_cfg_dict.get("eval_loss_fn") == "ssim" else L1Loss()

    @typecheck()
    def forward(
        self,
        y: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
        eta: torch.Tensor = None,
        hx: torch.Tensor = None,
        target: torch.Tensor = None,
        sigma: float = 1.0,
        **kwargs,
    ) -> Union[Generator, torch.Tensor]:
        """
        Forward pass of the network.
        Args:
            y: torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2], masked kspace data
            sensitivity_maps: torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2], coil sensitivity maps
            mask: torch.Tensor, shape [1, 1, n_x, n_y, 1], sampling mask
            eta: torch.Tensor, shape [batch_size, n_x, n_y, 2], initial guess for eta
            hx: torch.Tensor, shape [batch_size, n_x, n_y, 2], initial guess for hx
            target: torch.Tensor, shape [batch_size, n_x, n_y, 2], target data
            sigma: float, noise level
        Returns:
             Final estimation of the network.
             If self.accumulate_loss is True, returns a list of all intermediate estimates.
             If False, returns the final estimate.
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
                        f"`'initial_image` is required as input if initializer_initialization "
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

        eta = self.process_intermediate_eta(kspace_prediction, sensitivity_maps, target, do_coil_combination=True)
        return eta

    def process_intermediate_eta(self, eta, sensitivity_maps, target, do_coil_combination=False):
        """Process the intermediate eta to be used in the loss function."""
        # Take the last time step of the eta
        if do_coil_combination:
            eta = ifft2c(eta, fft_type=self.fft_type)
            eta = coil_combination(eta, sensitivity_maps, method=self.output_type, dim=1)
        eta = torch.view_as_complex(eta)
        _, eta = center_crop_to_smallest(target, eta)
        return eta

    @staticmethod
    def process_loss(target, eta, set_loss_fn):
        """Calculate the loss."""
        target = torch.abs(target / torch.max(torch.abs(target)))

        if "ssim" in str(set_loss_fn).lower():
            max_value = np.array(torch.max(torch.abs(target)).item()).astype(np.float32)

            def loss_fn(x, y):
                """Calculate the ssim loss."""
                return set_loss_fn(
                    x.unsqueeze(dim=1),
                    torch.abs(y / torch.max(torch.abs(y))).unsqueeze(dim=1),
                    data_range=torch.tensor(max_value).unsqueeze(dim=0).to(x.device),
                )

        else:

            def loss_fn(x, y):
                """Calculate other loss."""
                return set_loss_fn(x, torch.abs(y / torch.max(torch.abs(y))))

        return loss_fn(target, eta)

    @staticmethod
    def process_inputs(y, mask):
        """Process the inputs to the network."""
        if isinstance(y, list):
            r = np.random.randint(len(y))
            y = y[r]
            mask = mask[r]
        else:
            r = 0
        return y, mask, r

    def training_step(self, batch: Dict[float, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Training step for the RVM.
        Args:
            batch: A dictionary of the form {
                'y': subsampled kspace,
                'sensitivity_maps': sensitivity_maps,
                'mask': mask,
                'eta': initial estimation,
                'hx': hidden states,
                'target': target,
                'sigma': sigma
                }
            batch_idx: The index of the batch.
        Returns:
            A dictionary of the form {
                'loss': loss value,
                'log': log,
            }
        """
        y, sensitivity_maps, mask, _, target, _, _, acc, _, _ = batch
        y, mask, r = self.process_inputs(y, mask)
        etas = self.forward(y, sensitivity_maps, mask, None, None, target, 1.0)

        train_loss = self.process_loss(target, etas, set_loss_fn=self.train_loss_fn)

        acc = r if r != 0 else acc

        tensorboard_logs = {
            f"train_loss_{str(acc)}x": train_loss.item(),  # type: ignore
            "lr": self._optimizer.param_groups[0]["lr"],  # type: ignore
        }

        return {"loss": train_loss, "log": tensorboard_logs}

    def validation_step(self, batch: Dict[float, torch.Tensor], batch_idx: int) -> Dict:
        """Validation step for the RVM."""
        y, sensitivity_maps, mask, _, target, fname, slice_num, _, _, _ = batch
        y, mask, _ = self.process_inputs(y, mask)
        etas = self.forward(y, sensitivity_maps, mask, None, None, target, 1.0)

        val_loss = self.process_loss(target, etas, set_loss_fn=self.eval_loss_fn)

        if isinstance(etas, list):
            etas = etas[-1][-1]

        key = f"{fname[0]}_images_idx_{int(slice_num)}"  # type: ignore
        output = torch.abs(etas).detach().cpu()
        target = torch.abs(target).detach().cpu()
        output = output / output.max()  # type: ignore
        target = target / target.max()  # type: ignore
        error = torch.abs(target - output)
        self.log_image(f"{key}/target", target)
        self.log_image(f"{key}/reconstruction", output)
        self.log_image(f"{key}/error", error)

        return {
            "val_loss": val_loss,
        }

    def test_step(self, batch: Dict[float, torch.Tensor], batch_idx: int) -> Tuple[str, int, torch.Tensor]:
        """Test step for the RVM."""
        y, sensitivity_maps, mask, _, target, fname, slice_num, _, _, _ = batch
        y, mask, _ = self.process_inputs(y, mask)
        prediction = self.forward(y, sensitivity_maps, mask, None, None, target, 1.0)

        slice_num = int(slice_num)
        name = str(fname[0])  # type: ignore
        key = f"{name}_images_idx_{slice_num}"  # type: ignore

        output = torch.abs(prediction).detach().cpu()
        output = output / output.max()  # type: ignore

        target = torch.abs(target).detach().cpu()
        target = target / target.max()  # type: ignore

        error = torch.abs(target - output)

        self.log_image(f"{key}/target", target)
        self.log_image(f"{key}/reconstruction", output)
        self.log_image(f"{key}/error", error)

        return name, slice_num, prediction.detach().cpu().numpy()
