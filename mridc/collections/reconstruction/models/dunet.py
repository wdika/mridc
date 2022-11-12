# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from abc import ABC

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch.nn import L1Loss

import mridc.collections.common.losses.ssim as losses
import mridc.collections.common.parts.fft as fft
import mridc.collections.common.parts.utils as utils
import mridc.collections.reconstruction.models.base as base_models
import mridc.collections.reconstruction.models.didn.didn as didn_
import mridc.collections.reconstruction.models.sigmanet.dc_layers as dc_layers
import mridc.collections.reconstruction.models.sigmanet.sensitivity_net as sensitivity_net
import mridc.collections.reconstruction.models.unet_base.unet_block as unet_block
import mridc.core.classes.common as common_classes

__all__ = ["DUNet"]


class DUNet(base_models.BaseMRIReconstructionModel, ABC):
    """
    Implementation of the Down-Up NET, inspired by Hammernik, K, Schlemper, J, Qin, C, et al.

    References
    ----------

    ..

        Hammernik, K, Schlemper, J, Qin, C, et al. Systematic evaluation of iterative deep neural networks for fast \
        parallel MRI reconstruction with sensitivity-weighted coil combination. Magn Reson Med. 2021; 86: 1859– 1872. \
         https://doi.org/10.1002/mrm.28827

    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # init superclass
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        self.fft_centered = cfg_dict.get("fft_centered")
        self.fft_normalization = cfg_dict.get("fft_normalization")
        self.spatial_dims = cfg_dict.get("spatial_dims")
        self.coil_dim = cfg_dict.get("coil_dim")

        reg_model_architecture = cfg_dict.get("reg_model_architecture")
        if reg_model_architecture == "DIDN":
            reg_model = didn_.DIDN(
                in_channels=2,
                out_channels=2,
                hidden_channels=cfg_dict.get("didn_hidden_channels"),
                num_dubs=cfg_dict.get("didn_num_dubs"),
                num_convs_recon=cfg_dict.get("didn_num_convs_recon"),
            )
        elif reg_model_architecture in ["UNET", "NORMUNET"]:
            reg_model = unet_block.NormUnet(
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
                "DUNET is currently implemented for reg_model_architecture == 'DIDN' or 'UNet'."
                f"Got reg_model_architecture == {reg_model_architecture}."
            )

        data_consistency_term = cfg_dict.get("data_consistency_term")

        if data_consistency_term == "GD":
            dc_layer = dc_layers.DataGDLayer(
                lambda_init=cfg_dict.get("data_consistency_lambda_init"),
                fft_centered=self.fft_centered,
                fft_normalization=self.fft_normalization,
                spatial_dims=self.spatial_dims,
            )
        elif data_consistency_term == "PROX":
            dc_layer = dc_layers.DataProxCGLayer(
                lambda_init=cfg_dict.get("data_consistency_lambda_init"),
                fft_centered=self.fft_centered,
                fft_normalization=self.fft_normalization,
                spatial_dims=self.spatial_dims,
            )
        elif data_consistency_term == "VS":
            dc_layer = dc_layers.DataVSLayer(
                alpha_init=cfg_dict.get("data_consistency_alpha_init"),
                beta_init=cfg_dict.get("data_consistency_beta_init"),
                fft_centered=self.fft_centered,
                fft_normalization=self.fft_normalization,
                spatial_dims=self.spatial_dims,
            )
        else:
            dc_layer = dc_layers.DataIDLayer()

        self.model = sensitivity_net.SensitivityNetwork(
            cfg_dict.get("num_iter"),
            reg_model,
            dc_layer,
            shared_params=cfg_dict.get("shared_params"),
            save_space=False,
            reset_cache=False,
        )

        if cfg_dict.get("train_loss_fn") == "ssim":
            self.train_loss_fn = losses.SSIMLoss()
        elif cfg_dict.get("train_loss_fn") == "l1":
            self.train_loss_fn = L1Loss()
        elif cfg_dict.get("train_loss_fn") == "mse":
            self.train_loss_fn = torch.nn.MSELoss()
        else:
            raise ValueError("Unknown loss function: {}".format(cfg_dict.get("train_loss_fn")))
        if cfg_dict.get("eval_loss_fn") == "ssim":
            self.eval_loss_fn = losses.SSIMLoss()
        elif cfg_dict.get("eval_loss_fn") == "l1":
            self.eval_loss_fn = L1Loss()
        elif cfg_dict.get("eval_loss_fn") == "mse":
            self.eval_loss_fn = torch.nn.MSELoss()
        else:
            raise ValueError("Unknown loss function: {}".format(cfg_dict.get("eval_loss_fn")))

        self.dc_weight = torch.nn.Parameter(torch.ones(1))
        self.accumulate_estimates = False

    @common_classes.typecheck()
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
        init_pred = torch.sum(
            utils.complex_mul(
                fft.ifft2(
                    y, centered=self.fft_centered, normalization=self.fft_normalization, spatial_dims=self.spatial_dims
                ),
                utils.complex_conj(sensitivity_maps),
            ),
            self.coil_dim,
        )
        image = self.model(init_pred, y, sensitivity_maps, mask)
        image = torch.sum(utils.complex_mul(image, utils.complex_conj(sensitivity_maps)), self.coil_dim)
        image = torch.view_as_complex(image)
        _, image = utils.center_crop_to_smallest(target, image)
        return image
