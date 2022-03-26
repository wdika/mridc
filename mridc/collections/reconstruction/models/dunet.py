# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from abc import ABC

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch.nn import L1Loss

from mridc.collections.common.losses.ssim import SSIMLoss
from mridc.collections.common.parts.fft import ifft2c
from mridc.collections.common.parts.utils import complex_conj, complex_mul
from mridc.collections.reconstruction.models.base import BaseMRIReconstructionModel, BaseSensitivityModel
from mridc.collections.reconstruction.models.didn.didn import DIDN
from mridc.collections.reconstruction.models.sigmanet.dc_layers import (
    DataGDLayer,
    DataIDLayer,
    DataProxCGLayer,
    DataVSLayer,
)
from mridc.collections.reconstruction.models.sigmanet.sensitivity_net import SensitivityNetwork
from mridc.collections.reconstruction.models.unet_base.unet_block import NormUnet
from mridc.collections.reconstruction.parts.utils import center_crop_to_smallest
from mridc.core.classes.common import typecheck

__all__ = ["DUNet"]


class DUNet(BaseMRIReconstructionModel, ABC):
    """
    Based on the DUNET implementation [1]_.

    References
    ----------
    .. [1]  Hammernik, K, Schlemper, J, Qin, C, et al. Systematic evaluation of iterative deep neural networks for
    fast parallel MRI reconstruction with sensitivity-weighted coil combination. Magn Reson Med. 2021; 86: 1859– 1872.
    https://doi.org/10.1002/mrm.28827
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # init superclass
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        self.fft_type = cfg_dict.get("fft_type")

        reg_model_architecture = cfg_dict.get("reg_model_architecture")
        if reg_model_architecture == "DIDN":
            reg_model = DIDN(
                in_channels=2,
                out_channels=2,
                hidden_channels=cfg_dict.get("didn_hidden_channels"),
                num_dubs=cfg_dict.get("didn_num_dubs"),
                num_convs_recon=cfg_dict.get("didn_num_convs_recon"),
            )
        elif reg_model_architecture in ["UNET", "NORMUNET"]:
            reg_model = NormUnet(
                cfg_dict.get("unet_num_filters"),
                cfg_dict.get("unet_num_pool_layers"),
                in_chans=2,
                out_chans=2,
                drop_prob=cfg_dict.get("unet_dropout_probability"),
                padding_size=cfg_dict.get("unet_padding_size"),
                normalize=cfg_dict.get("unet_normalize"),
            )
        else:
            raise NotImplementedError(
                f"DUNET is currently implemented for reg_model_architecture == 'DIDN' or 'UNet'."
                f"Got reg_model_architecture == {reg_model_architecture}."
            )

        data_consistency_term = cfg_dict.get("data_consistency_term")

        if data_consistency_term == "GD":
            dc_layer = DataGDLayer(lambda_init=cfg_dict.get("data_consistency_lambda_init"), fft_type=self.fft_type)
        elif data_consistency_term == "PROX":
            dc_layer = DataProxCGLayer(
                lambda_init=cfg_dict.get("data_consistency_lambda_init"), fft_type=self.fft_type
            )
        elif data_consistency_term == "VS":
            dc_layer = DataVSLayer(
                alpha_init=cfg_dict.get("data_consistency_alpha_init"),
                beta_init=cfg_dict.get("data_consistency_beta_init"),
                fft_type=self.fft_type,
            )
        else:
            dc_layer = DataIDLayer()

        self.model = SensitivityNetwork(
            cfg_dict.get("num_iter"),
            reg_model,
            dc_layer,
            shared_params=cfg_dict.get("shared_params"),
            save_space=False,
            reset_cache=False,
        )

        self._coil_dim = 1

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

        self.train_loss_fn = SSIMLoss() if cfg_dict.get("train_loss_fn") == "ssim" else L1Loss()
        self.eval_loss_fn = SSIMLoss() if cfg_dict.get("eval_loss_fn") == "ssim" else L1Loss()
        self.output_type = cfg_dict.get("output_type")

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
        init_pred = torch.sum(complex_mul(ifft2c(y, fft_type=self.fft_type), complex_conj(sensitivity_maps)), 1)
        image = self.model(init_pred, y, sensitivity_maps, mask)
        image = torch.sum(complex_mul(image, complex_conj(sensitivity_maps)), 1)
        image = torch.view_as_complex(image)
        _, image = center_crop_to_smallest(target, image)
        return image
