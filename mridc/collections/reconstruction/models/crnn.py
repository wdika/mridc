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
from mridc.collections.common.parts.fft import ifft2
from mridc.collections.common.parts.utils import coil_combination
from mridc.collections.reconstruction.models.base import BaseMRIReconstructionModel
from mridc.collections.reconstruction.models.conv.gruconv2d import GRUConv2d
from mridc.collections.reconstruction.models.convrecnet.crnn_block import RecurrentConvolutionalNetBlock
from mridc.collections.reconstruction.parts.utils import center_crop_to_smallest
from mridc.core.classes.common import typecheck

__all__ = ["CRNNet"]


class CRNNet(BaseMRIReconstructionModel, ABC):
    """
    Implementation of the Convolutional Recurrent Neural Network, inspired by C. Qin, J. Schlemper, J. Caballero, \
    A. N. Price, J. V. Hajnal and D. Rueckert.

    References
    ----------

    ..

        C. Qin, J. Schlemper, J. Caballero, A. N. Price, J. V. Hajnal and D. Rueckert, "Convolutional Recurrent \
        Neural Networks for Dynamic MR Image Reconstruction," in IEEE Transactions on Medical Imaging, vol. 38, \
        no. 1, pp. 280-290, Jan. 2019, doi: 10.1109/TMI.2018.2863670.

    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # init superclass
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.no_dc = cfg_dict.get("no_dc")
        self.fft_centered = cfg_dict.get("fft_centered")
        self.fft_normalization = cfg_dict.get("fft_normalization")
        self.spatial_dims = cfg_dict.get("spatial_dims")
        self.coil_dim = cfg_dict.get("coil_dim")
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
            fft_centered=self.fft_centered,
            fft_normalization=self.fft_normalization,
            spatial_dims=self.spatial_dims,
            coil_dim=self.coil_dim,
            no_dc=self.no_dc,
        )

        self.coil_combination_method = cfg_dict.get("coil_combination_method")

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
        pred = self.crnn(y, sensitivity_maps, mask)
        yield [self.process_intermediate_pred(x, sensitivity_maps, target) for x in pred]

    def process_intermediate_pred(self, pred, sensitivity_maps, target):
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

        Returns
        -------
        pred: torch.Tensor, shape [batch_size, n_x, n_y, 2]
            Processed prediction.
        """
        pred = ifft2(
            pred, centered=self.fft_centered, normalization=self.fft_normalization, spatial_dims=self.spatial_dims
        )
        pred = coil_combination(pred, sensitivity_maps, method=self.coil_combination_method, dim=self.coil_dim)
        pred = torch.view_as_complex(pred)
        _, pred = center_crop_to_smallest(target, pred)
        return pred

    def process_loss(self, target, pred, _loss_fn):
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
                return _loss_fn(
                    x.unsqueeze(dim=self.coil_dim),
                    torch.abs(y / torch.max(torch.abs(y))).unsqueeze(dim=self.coil_dim),
                    data_range=torch.tensor(max_value).unsqueeze(dim=0).to(x.device),
                )

        else:

            def loss_fn(x, y):
                """Calculate other loss."""
                return _loss_fn(x, torch.abs(y / torch.max(torch.abs(y))))

        iterations_loss = [loss_fn(target, iteration_pred) for iteration_pred in pred]
        _loss = [x * torch.logspace(-1, 0, steps=self.num_iterations).to(iterations_loss[0]) for x in iterations_loss]
        yield sum(sum(_loss) / self.num_iterations)
