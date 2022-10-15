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
import mridc.collections.reconstruction.models.conv.conv2d as conv2d
import mridc.collections.reconstruction.models.didn.didn as didn_
import mridc.collections.reconstruction.models.mwcnn.mwcnn as mwcnn_
import mridc.collections.reconstruction.models.primaldual.pd as pd
import mridc.collections.reconstruction.models.unet_base.unet_block as unet_block
import mridc.core.classes.common as common_classes

__all__ = ["LPDNet"]


class LPDNet(base_models.BaseMRIReconstructionModel, ABC):
    """
    Implementation of the Learned Primal Dual network, inspired by Adler, Jonas, and Ozan Öktem.

    References
    ----------

    ..

        Adler, Jonas, and Ozan Öktem. “Learned Primal-Dual Reconstruction.” IEEE Transactions on Medical Imaging, \
        vol. 37, no. 6, June 2018, pp. 1322–32. arXiv.org, https://doi.org/10.1109/TMI.2018.2799231.

    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # init superclass
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.num_iter = cfg_dict.get("num_iter")
        self.num_primal = cfg_dict.get("num_primal")
        self.num_dual = cfg_dict.get("num_dual")

        primal_model_architecture = cfg_dict.get("primal_model_architecture")

        if primal_model_architecture == "MWCNN":
            primal_model = torch.nn.Sequential(
                *[
                    mwcnn_.MWCNN(
                        input_channels=2 * (self.num_primal + 1),
                        first_conv_hidden_channels=cfg_dict.get("primal_mwcnn_hidden_channels"),
                        num_scales=cfg_dict.get("primal_mwcnn_num_scales"),
                        bias=cfg_dict.get("primal_mwcnn_bias"),
                        batchnorm=cfg_dict.get("primal_mwcnn_batchnorm"),
                    ),
                    torch.nn.Conv2d(2 * (self.num_primal + 1), 2 * self.num_primal, kernel_size=1),
                ]
            )
        elif primal_model_architecture in ["UNET", "NORMUNET"]:
            primal_model = unet_block.NormUnet(
                cfg_dict.get("primal_unet_num_filters"),
                cfg_dict.get("primal_unet_num_pool_layers"),
                in_chans=2 * (self.num_primal + 1),
                out_chans=2 * self.num_primal,
                drop_prob=cfg_dict.get("primal_unet_dropout_probability"),
                padding_size=cfg_dict.get("primal_unet_padding_size"),
                normalize=cfg_dict.get("primal_unet_normalize"),
            )
        else:
            raise NotImplementedError(
                "LPDNet is currently implemented for primal_model_architecture == 'CONV' or 'UNet'."
                f"Got primal_model_architecture == {primal_model_architecture}."
            )

        dual_model_architecture = cfg_dict.get("dual_model_architecture")

        if dual_model_architecture == "CONV":
            dual_model = conv2d.Conv2d(
                in_channels=2 * (self.num_dual + 2),
                out_channels=2 * self.num_dual,
                hidden_channels=cfg_dict.get("kspace_conv_hidden_channels"),
                n_convs=cfg_dict.get("kspace_conv_n_convs"),
                batchnorm=cfg_dict.get("kspace_conv_batchnorm"),
            )
        elif dual_model_architecture == "DIDN":
            dual_model = didn_.DIDN(
                in_channels=2 * (self.num_dual + 2),
                out_channels=2 * self.num_dual,
                hidden_channels=cfg_dict.get("kspace_didn_hidden_channels"),
                num_dubs=cfg_dict.get("kspace_didn_num_dubs"),
                num_convs_recon=cfg_dict.get("kspace_didn_num_convs_recon"),
            )
        elif dual_model_architecture in ["UNET", "NORMUNET"]:
            dual_model = unet_block.NormUnet(
                cfg_dict.get("dual_unet_num_filters"),
                cfg_dict.get("dual_unet_num_pool_layers"),
                in_chans=2 * (self.num_dual + 2),
                out_chans=2 * self.num_dual,
                drop_prob=cfg_dict.get("dual_unet_dropout_probability"),
                padding_size=cfg_dict.get("dual_unet_padding_size"),
                normalize=cfg_dict.get("dual_unet_normalize"),
            )
        else:
            raise NotImplementedError(
                "LPDNet is currently implemented for dual_model_architecture == 'CONV' or 'DIDN' or 'UNet'."
                f"Got dual_model_architecture == {dual_model_architecture}."
            )

        self.primal_net = torch.nn.ModuleList(
            [pd.PrimalNet(self.num_primal, primal_architecture=primal_model) for _ in range(self.num_iter)]
        )
        self.dual_net = torch.nn.ModuleList(
            [pd.DualNet(self.num_dual, dual_architecture=dual_model) for _ in range(self.num_iter)]
        )

        self.fft_centered = cfg_dict.get("fft_centered")
        self.fft_normalization = cfg_dict.get("fft_normalization")
        self.spatial_dims = cfg_dict.get("spatial_dims")
        self.coil_dim = cfg_dict.get("coil_dim")

        self.train_loss_fn = losses.SSIMLoss() if cfg_dict.get("train_loss_fn") == "ssim" else L1Loss()
        self.eval_loss_fn = losses.SSIMLoss() if cfg_dict.get("eval_loss_fn") == "ssim" else L1Loss()

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
        input_image = utils.complex_mul(
            fft.ifft2(
                torch.where(mask == 0, torch.tensor([0.0], dtype=y.dtype).to(y.device), y),
                centered=self.fft_centered,
                normalization=self.fft_normalization,
                spatial_dims=self.spatial_dims,
            ),
            utils.complex_conj(sensitivity_maps),
        ).sum(self.coil_dim)
        dual_buffer = torch.cat([y] * self.num_dual, -1).to(y.device)
        primal_buffer = torch.cat([input_image] * self.num_primal, -1).to(y.device)

        for idx in range(self.num_iter):
            # Dual
            f_2 = primal_buffer[..., 2:4].clone()
            f_2 = torch.where(
                mask == 0,
                torch.tensor([0.0], dtype=f_2.dtype).to(f_2.device),
                fft.fft2(
                    utils.complex_mul(f_2.unsqueeze(self.coil_dim), sensitivity_maps),
                    centered=self.fft_centered,
                    normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                ).type(f_2.type()),
            )
            dual_buffer = self.dual_net[idx](dual_buffer, f_2, y)

            # Primal
            h_1 = dual_buffer[..., 0:2].clone()
            # needed for python3.9
            h_1 = torch.view_as_real(h_1[..., 0] + 1j * h_1[..., 1])
            h_1 = utils.complex_mul(
                fft.ifft2(
                    torch.where(mask == 0, torch.tensor([0.0], dtype=h_1.dtype).to(h_1.device), h_1),
                    centered=self.fft_centered,
                    normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                ),
                utils.complex_conj(sensitivity_maps),
            ).sum(self.coil_dim)
            primal_buffer = self.primal_net[idx](primal_buffer, h_1)

        output = primal_buffer[..., 0:2]
        output = (output**2).sum(-1).sqrt()
        _, output = utils.center_crop_to_smallest(target, output)
        return output
