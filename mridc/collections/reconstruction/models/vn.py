# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from abc import ABC

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch.nn import L1Loss

from mridc.collections.common.losses.ssim import SSIMLoss
from mridc.collections.common.parts.fft import ifft2c
from mridc.collections.common.parts.utils import coil_combination
from mridc.collections.reconstruction.models.base import BaseMRIReconstructionModel, BaseSensitivityModel
from mridc.collections.reconstruction.models.unet_base.unet_block import NormUnet
from mridc.collections.reconstruction.models.varnet.vn_block import VarNetBlock
from mridc.collections.reconstruction.parts.utils import center_crop_to_smallest
from mridc.core.classes.common import typecheck

__all__ = ["VarNet"]


class VarNet(BaseMRIReconstructionModel, ABC):
    """
    Implementation of the End-to-end Variational Network (VN), as presented in Sriram, A. et al.

    References
    ----------

    ..

        Sriram, A. et al. (2020) ‘End-to-End Variational Networks for Accelerated MRI Reconstruction’. Available \
        at: https://github.com/facebookresearch/fastMRI.

    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # init superclass
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.no_dc = cfg_dict.get("no_dc")
        self.fft_type = cfg_dict.get("fft_type")
        self.num_cascades = cfg_dict.get("num_cascades")

        # Cascades of VN blocks
        self.cascades = torch.nn.ModuleList(
            [
                VarNetBlock(
                    NormUnet(
                        chans=cfg_dict.get("channels"),
                        num_pools=cfg_dict.get("pooling_layers"),
                        padding_size=cfg_dict.get("padding_size"),
                        normalize=cfg_dict.get("normalize"),
                    ),
                    fft_type=self.fft_type,
                    no_dc=self.no_dc,
                )
                for _ in range(self.num_cascades)
            ]
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

        # initialize weights if not using pretrained vn
        # TODO if not cfg_dict.get("pretrained", False)

        self.train_loss_fn = SSIMLoss() if cfg_dict.get("train_loss_fn") == "ssim" else L1Loss()
        self.eval_loss_fn = SSIMLoss() if cfg_dict.get("eval_loss_fn") == "ssim" else L1Loss()

        self.dc_weight = torch.nn.Parameter(torch.ones(1))
        self.accumulate_estimates = False

    @typecheck()
    def forward(
        self,
        y: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
        init_pred: torch.Tensor,
        target: torch.Tensor,
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
        sensitivity_maps = self.sens_net(y, mask) if self.use_sens_net else sensitivity_maps
        estimation = y.clone()

        for cascade in self.cascades:
            # Forward pass through the cascades
            estimation = cascade(estimation, y, sensitivity_maps, mask)

        estimation = ifft2c(estimation, fft_type=self.fft_type)
        estimation = coil_combination(estimation, sensitivity_maps, method=self.output_type, dim=1)
        estimation = torch.view_as_complex(estimation)
        _, estimation = center_crop_to_smallest(target, estimation)
        return estimation
