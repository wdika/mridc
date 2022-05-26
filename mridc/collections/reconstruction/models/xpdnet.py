# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from abc import ABC

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch.nn import L1Loss

from mridc.collections.common.losses.ssim import SSIMLoss
from mridc.collections.reconstruction.models.base import BaseMRIReconstructionModel
from mridc.collections.reconstruction.models.conv.conv2d import Conv2d
from mridc.collections.reconstruction.models.crossdomain.crossdomain import CrossDomainNetwork
from mridc.collections.reconstruction.models.crossdomain.multicoil import MultiCoil
from mridc.collections.reconstruction.models.didn.didn import DIDN
from mridc.collections.reconstruction.models.mwcnn.mwcnn import MWCNN
from mridc.collections.reconstruction.models.unet_base.unet_block import NormUnet
from mridc.collections.reconstruction.parts.utils import center_crop_to_smallest
from mridc.core.classes.common import typecheck

__all__ = ["XPDNet"]


class XPDNet(BaseMRIReconstructionModel, ABC):
    """
    Implementation of the XPDNet, as presented in Ramzi, Zaccharie, et al.

    References
    ----------

    ..

        Ramzi, Zaccharie, et al. “XPDNet for MRI Reconstruction: An Application to the 2020 FastMRI Challenge. \
        ” ArXiv:2010.07290 [Physics, Stat], July 2021. arXiv.org, http://arxiv.org/abs/2010.07290.

    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # init superclass
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        num_primal = cfg_dict.get("num_primal")
        num_dual = cfg_dict.get("num_dual")
        num_iter = cfg_dict.get("num_iter")

        kspace_model_architecture = cfg_dict.get("kspace_model_architecture")
        dual_conv_hidden_channels = cfg_dict.get("dual_conv_hidden_channels")
        dual_conv_num_dubs = cfg_dict.get("dual_conv_num_dubs")
        dual_conv_batchnorm = cfg_dict.get("dual_conv_batchnorm")
        dual_didn_hidden_channels = cfg_dict.get("dual_didn_hidden_channels")
        dual_didn_num_dubs = cfg_dict.get("dual_didn_num_dubs")
        dual_didn_num_convs_recon = cfg_dict.get("dual_didn_num_convs_recon")

        if cfg_dict.get("use_primal_only"):
            kspace_model_list = None
            num_dual = 1
        elif kspace_model_architecture == "CONV":
            kspace_model_list = torch.nn.ModuleList(
                [
                    MultiCoil(
                        Conv2d(
                            2 * (num_dual + num_primal + 1),
                            2 * num_dual,
                            dual_conv_hidden_channels,
                            dual_conv_num_dubs,
                            batchnorm=dual_conv_batchnorm,
                        )
                    )
                    for _ in range(num_iter)
                ]
            )
        elif kspace_model_architecture == "DIDN":
            kspace_model_list = torch.nn.ModuleList(
                [
                    MultiCoil(
                        DIDN(
                            in_channels=2 * (num_dual + num_primal + 1),
                            out_channels=2 * num_dual,
                            hidden_channels=dual_didn_hidden_channels,
                            num_dubs=dual_didn_num_dubs,
                            num_convs_recon=dual_didn_num_convs_recon,
                        )
                    )
                    for _ in range(num_iter)
                ]
            )
        elif kspace_model_architecture in ["UNET", "NORMUNET"]:
            kspace_model_list = torch.nn.ModuleList(
                [
                    MultiCoil(
                        NormUnet(
                            cfg_dict.get("kspace_unet_num_filters"),
                            cfg_dict.get("kspace_unet_num_pool_layers"),
                            in_chans=2 * (num_dual + num_primal + 1),
                            out_chans=2 * num_dual,
                            drop_prob=cfg_dict.get("kspace_unet_dropout_probability"),
                            padding_size=cfg_dict.get("kspace_unet_padding_size"),
                            normalize=cfg_dict.get("kspace_unet_normalize"),
                        ),
                        coil_to_batch=True,
                    )
                    for _ in range(num_iter)
                ]
            )
        else:
            raise NotImplementedError(
                "XPDNet is currently implemented for kspace_model_architecture == 'CONV' or 'DIDN'."
                f"Got kspace_model_architecture == {kspace_model_architecture}."
            )

        image_model_architecture = cfg_dict.get("image_model_architecture")
        mwcnn_hidden_channels = cfg_dict.get("mwcnn_hidden_channels")
        mwcnn_num_scales = cfg_dict.get("mwcnn_num_scales")
        mwcnn_bias = cfg_dict.get("mwcnn_bias")
        mwcnn_batchnorm = cfg_dict.get("mwcnn_batchnorm")

        if image_model_architecture == "MWCNN":
            image_model_list = torch.nn.ModuleList(
                [
                    torch.nn.Sequential(
                        MWCNN(
                            input_channels=2 * (num_primal + num_dual),
                            first_conv_hidden_channels=mwcnn_hidden_channels,
                            num_scales=mwcnn_num_scales,
                            bias=mwcnn_bias,
                            batchnorm=mwcnn_batchnorm,
                        ),
                        torch.nn.Conv2d(2 * (num_primal + num_dual), 2 * num_primal, kernel_size=3, padding=1),
                    )
                    for _ in range(num_iter)
                ]
            )
        elif image_model_architecture in ["UNET", "NORMUNET"]:
            image_model_list = torch.nn.ModuleList(
                [
                    NormUnet(
                        cfg_dict.get("imspace_unet_num_filters"),
                        cfg_dict.get("imspace_unet_num_pool_layers"),
                        in_chans=2 * (num_primal + num_dual),
                        out_chans=2 * num_primal,
                        drop_prob=cfg_dict.get("imspace_unet_dropout_probability"),
                        padding_size=cfg_dict.get("imspace_unet_padding_size"),
                        normalize=cfg_dict.get("imspace_unet_normalize"),
                    )
                    for _ in range(num_iter)
                ]
            )
        else:
            raise NotImplementedError(f"Image model architecture {image_model_architecture} not found for XPDNet.")

        self.fft_normalization = cfg_dict.get("fft_normalization")
        self.spatial_dims = cfg_dict.get("spatial_dims")
        self.coil_dim = cfg_dict.get("coil_dim")
        self.num_cascades = cfg_dict.get("num_cascades")

        self.xpdnet = CrossDomainNetwork(
            image_model_list=image_model_list,
            kspace_model_list=kspace_model_list,
            domain_sequence="KI" * num_iter,
            image_buffer_size=num_primal,
            kspace_buffer_size=num_dual,
            normalize_image=cfg_dict.get("normalize_image"),
            fft_centered=self.fft_centered,
            fft_normalization=self.fft_normalization,
            spatial_dims=self.spatial_dims,
            coil_dim=self.coil_dim,
        )

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
        eta = self.xpdnet(y, sensitivity_maps, mask)
        eta = (eta**2).sqrt().sum(-1)
        _, eta = center_crop_to_smallest(target, eta)
        return eta
