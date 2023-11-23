# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from abc import ABC

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

import mridc.collections.reconstruction.nn.base as base_models
import mridc.collections.reconstruction.nn.crossdomain.multicoil as crossdomain_multicoil
import mridc.collections.reconstruction.nn.didn.didn as didn_
import mridc.collections.reconstruction.nn.mwcnn.mwcnn as mwcnn_
import mridc.core.classes.common as common_classes
from mridc.collections.common.parts import utils
from mridc.collections.reconstruction.nn.conv import conv2d
from mridc.collections.reconstruction.nn.crossdomain import crossdomain
from mridc.collections.reconstruction.nn.unet_base import unet_block

__all__ = ["XPDNet"]


class XPDNet(base_models.BaseMRIReconstructionModel, ABC):  # type: ignore
    """
    Implementation of the XPDNet, as presented in [1].

    References
    ----------
    .. [1] Ramzi, Zaccharie, et al. “XPDNet for MRI Reconstruction: An Application to the 2020 FastMRI Challenge.
        ArXiv:2010.07290 [Physics, Stat], July 2021. arXiv.org, http://arxiv.org/abs/2010.07290.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):  # noqa: W0221
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        num_primal = cfg_dict.get("num_primal")
        num_dual = cfg_dict.get("num_dual")
        num_iter = cfg_dict.get("num_iter")

        kspace_model_architecture = cfg_dict.get("kspace_model_architecture")
        dual_conv_hidden_channels = cfg_dict.get("dual_conv_hidden_channels", 64)
        dual_conv_num_dubs = cfg_dict.get("dual_conv_num_dubs", 2)
        dual_conv_batchnorm = cfg_dict.get("dual_conv_batchnorm", True)
        dual_didn_hidden_channels = cfg_dict.get("dual_didn_hidden_channels", 64)
        dual_didn_num_dubs = cfg_dict.get("dual_didn_num_dubs", 2)
        dual_didn_num_convs_recon = cfg_dict.get("dual_didn_num_convs_recon", True)

        if cfg_dict.get("use_primal_only"):
            kspace_model_list = None
            num_dual = 1
        elif kspace_model_architecture == "CONV":
            kspace_model_list = torch.nn.ModuleList(
                [
                    crossdomain_multicoil.MultiCoil(
                        conv2d.Conv2d(
                            cfg_dict.get("kspace_in_channels") * (num_dual + num_primal + 1),
                            cfg_dict.get("kspace_out_channels") * num_dual,
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
                    crossdomain_multicoil.MultiCoil(
                        didn_.DIDN(
                            in_channels=cfg_dict.get("kspace_in_channels") * (num_dual + num_primal + 1),
                            out_channels=cfg_dict.get("kspace_out_channels") * num_dual,
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
                    crossdomain_multicoil.MultiCoil(
                        unet_block.NormUnet(
                            cfg_dict.get("kspace_unet_num_filters"),
                            cfg_dict.get("kspace_unet_num_pool_layers"),
                            in_chans=cfg_dict.get("kspace_in_channels") * (num_dual + num_primal + 1),
                            out_chans=cfg_dict.get("kspace_out_channels") * num_dual,
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
        mwcnn_hidden_channels = cfg_dict.get("mwcnn_hidden_channels", 16)
        mwcnn_num_scales = cfg_dict.get("mwcnn_num_scales", 2)
        mwcnn_bias = cfg_dict.get("mwcnn_bias", True)
        mwcnn_batchnorm = cfg_dict.get("mwcnn_batchnorm", True)

        if image_model_architecture == "MWCNN":
            image_model_list = torch.nn.ModuleList(
                [
                    torch.nn.Sequential(
                        mwcnn_.MWCNN(
                            input_channels=cfg_dict.get("imspace_in_channels") * (num_primal + num_dual),
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
                    unet_block.NormUnet(
                        cfg_dict.get("imspace_unet_num_filters"),
                        cfg_dict.get("imspace_unet_num_pool_layers"),
                        in_chans=cfg_dict.get("imspace_in_channels") * (num_primal + num_dual),
                        out_chans=cfg_dict.get("imspace_out_channels") * num_primal,
                        drop_prob=cfg_dict.get("imspace_unet_dropout_probability"),
                        padding_size=cfg_dict.get("imspace_unet_padding_size"),
                        normalize=cfg_dict.get("imspace_unet_normalize"),
                    )
                    for _ in range(num_iter)
                ]
            )
        else:
            raise NotImplementedError(f"Image model architecture {image_model_architecture} not found for XPDNet.")

        self.num_cascades = cfg_dict.get("num_cascades")

        self.xpdnet = crossdomain.CrossDomainNetwork(
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

    @common_classes.typecheck()  # type: ignore
    def forward(  # noqa: W0221
        self,
        y: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
        init_pred: torch.Tensor,  # noqa: W0613
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the network.

        Parameters
        ----------
        y : torch.Tensor
            Subsampled k-space data. Shape [batch_size, n_coils, n_x, n_y, 2]
        sensitivity_maps : torch.Tensor
            Coil sensitivity maps. Shape [batch_size, n_coils, n_x, n_y, 2]
        mask : torch.Tensor
            Subsampling mask. Shape [1, 1, n_x, n_y, 1]
        init_pred : torch.Tensor
            Initial prediction. Shape [batch_size, n_x, n_y, 2]
        target : torch.Tensor
            Target data to compute the loss. Shape [batch_size, n_x, n_y, 2]

        Returns
        -------
        torch.Tensor
            Reconstructed image. Shape [batch_size, n_x, n_y, 2]
        """
        prediction = self.xpdnet(y, sensitivity_maps, mask)
        prediction = (prediction**2).sqrt().sum(-1)
        if target.shape[-1] == 2:
            target = torch.view_as_complex(target)
        _, prediction = utils.center_crop_to_smallest(target, prediction)
        return prediction
