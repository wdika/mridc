# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from abc import ABC
from typing import Generator, Union

import numpy as np
import torch

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch.nn import L1Loss

from mridc.collections.common.losses.ssim import SSIMLoss
from mridc.collections.common.parts.fft import ifft2c
from mridc.collections.common.parts.utils import coil_combination
from mridc.collections.reconstruction.models.base import BaseMRIReconstructionModel, BaseSensitivityModel
from mridc.collections.reconstruction.models.conv.gruconv2d import GRUConv2d
from mridc.collections.reconstruction.models.convrecnet.crnn_block import RecurrentConvolutionalNetBlock
from mridc.collections.reconstruction.parts.utils import center_crop_to_smallest
from mridc.core.classes.common import typecheck

__all__ = ["CRNNet"]


class CRNNet(BaseMRIReconstructionModel, ABC):
    """
    Convolutional Recurrent Neural Network implementation inspired by [1]_.

    References
    ----------
    .. [1] C. Qin, J. Schlemper, J. Caballero, A. N. Price, J. V. Hajnal and D. Rueckert,
    "Convolutional Recurrent Neural Networks for Dynamic MR Image Reconstruction," in IEEE Transactions on Medical
    Imaging, vol. 38, no. 1, pp. 280-290, Jan. 2019, doi: 10.1109/TMI.2018.2863670.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # init superclass
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.no_dc = cfg_dict.get("no_dc")
        self.fft_type = cfg_dict.get("fft_type")
        self.num_iterations = cfg_dict.get("num_iterations")

        self.crnn = RecurrentConvolutionalNetBlock(
            GRUConv2d(
                in_channels=2,
                out_channels=2,
                hidden_channels=cfg_dict.get("hidden_channels"),
                n_convs=cfg_dict.get("n_convs"),
                batchnorm=cfg_dict.get("batchnorm"),
            ),
            num_iterations=self.num_iterations,
            fft_type=self.fft_type,
            no_dc=self.no_dc,
        )

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
            )

        # initialize weights if not using pretrained ccnn
        # TODO if not ccnn_cfg_dict.get("pretrained", False)

        self.train_loss_fn = SSIMLoss() if cfg_dict.get("train_loss_fn") == "ssim" else L1Loss()
        self.eval_loss_fn = SSIMLoss() if cfg_dict.get("eval_loss_fn") == "ssim" else L1Loss()

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
        pred = self.crnn(y, sensitivity_maps, mask)
        yield [self.process_intermediate_eta(x, sensitivity_maps, target) for x in pred]

    def process_intermediate_eta(self, eta, sensitivity_maps, target):
        """Process the intermediate eta to be used in the loss function."""
        eta = ifft2c(eta, fft_type=self.fft_type)
        eta = coil_combination(eta, sensitivity_maps, method=self.output_type, dim=1)
        eta = torch.view_as_complex(eta)
        _, eta = center_crop_to_smallest(target, eta)
        return eta

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

        iterations_loss = [loss_fn(target, iteration_eta) for iteration_eta in eta]
        _loss = [x * torch.logspace(-1, 0, steps=self.num_iterations).to(iterations_loss[0]) for x in iterations_loss]
        yield sum(sum(_loss) / self.num_iterations)
