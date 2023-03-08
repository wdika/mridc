# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from abc import ABC

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

import mridc.collections.reconstruction.nn.base as base_models
import mridc.collections.reconstruction.nn.crossdomain.multicoil as crossdomain
import mridc.collections.reconstruction.nn.didn.didn as didn_
import mridc.collections.reconstruction.nn.mwcnn.mwcnn as mwcnn_
import mridc.core.classes.common as common_classes
from mridc.collections.common.parts import fft, utils
from mridc.collections.reconstruction.nn.conv import conv2d
from mridc.collections.reconstruction.nn.unet_base import unet_block

__all__ = ["KIKINet"]


class KIKINet(base_models.BaseMRIReconstructionModel, ABC):  # type: ignore
    """
    Based on KIKINet implementation. Modified to work with multi-coil k-space data, as presented in [1].

    References
    ----------
    .. [1] Eo, Taejoon, et al. “KIKI-Net: Cross-Domain Convolutional Neural Networks for Reconstructing Undersampled
        Magnetic Resonance Images.” Magnetic Resonance in Medicine, vol. 80, no. 5, Nov. 2018, pp. 2188–201. PubMed,
        https://doi.org/10.1002/mrm.27201.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.num_iter = cfg_dict.get("num_iter")
        self.no_dc = cfg_dict.get("no_dc")

        kspace_model_architecture = cfg_dict.get("kspace_model_architecture")

        if kspace_model_architecture == "CONV":
            kspace_model = conv2d.Conv2d(
                in_channels=cfg_dict.get("kspace_in_channels", 2),
                out_channels=cfg_dict.get("kspace_out_channels", 2),
                hidden_channels=cfg_dict.get("kspace_conv_hidden_channels"),
                n_convs=cfg_dict.get("kspace_conv_n_convs"),
                batchnorm=cfg_dict.get("kspace_conv_batchnorm"),
            )
        elif kspace_model_architecture == "DIDN":
            kspace_model = didn_.DIDN(
                in_channels=cfg_dict.get("kspace_in_channels", 2),
                out_channels=cfg_dict.get("kspace_out_channels", 2),
                hidden_channels=cfg_dict.get("kspace_didn_hidden_channels"),
                num_dubs=cfg_dict.get("kspace_didn_num_dubs"),
                num_convs_recon=cfg_dict.get("kspace_didn_num_convs_recon"),
            )
        elif kspace_model_architecture in ["UNET", "NORMUNET"]:
            kspace_model = unet_block.NormUnet(
                cfg_dict.get("kspace_unet_num_filters"),
                cfg_dict.get("kspace_unet_num_pool_layers"),
                in_chans=cfg_dict.get("kspace_in_channels", 2),
                out_chans=cfg_dict.get("kspace_out_channels", 2),
                drop_prob=cfg_dict.get("kspace_unet_dropout_probability"),
                padding_size=cfg_dict.get("kspace_unet_padding_size"),
                normalize=cfg_dict.get("kspace_unet_normalize"),
            )
        else:
            raise NotImplementedError(
                "KIKINet is currently implemented for kspace_model_architecture == 'CONV' or 'DIDN' or 'UNet'."
                f"Got kspace_model_architecture == {kspace_model_architecture}."
            )

        image_model_architecture = cfg_dict.get("imspace_model_architecture")

        if image_model_architecture == "MWCNN":
            image_model = mwcnn_.MWCNN(
                input_channels=cfg_dict.get("imspace_in_channels", 2),
                first_conv_hidden_channels=cfg_dict.get("image_mwcnn_hidden_channels"),
                num_scales=cfg_dict.get("image_mwcnn_num_scales"),
                bias=cfg_dict.get("image_mwcnn_bias"),
                batchnorm=cfg_dict.get("image_mwcnn_batchnorm"),
            )
        elif image_model_architecture in ["UNET", "NORMUNET"]:
            image_model = unet_block.NormUnet(
                cfg_dict.get("imspace_unet_num_filters"),
                cfg_dict.get("imspace_unet_num_pool_layers"),
                in_chans=cfg_dict.get("imspace_in_channels", 2),
                out_chans=cfg_dict.get("imspace_out_channels", 2),
                drop_prob=cfg_dict.get("imspace_unet_dropout_probability"),
                padding_size=cfg_dict.get("imspace_unet_padding_size"),
                normalize=cfg_dict.get("imspace_unet_normalize"),
            )
        else:
            raise NotImplementedError(
                "KIKINet is currently implemented only with image_model_architecture == 'MWCNN' or 'UNet'."
                f"Got {image_model_architecture}."
            )

        self.image_model_list = torch.nn.ModuleList([image_model] * self.num_iter)
        self.kspace_model_list = torch.nn.ModuleList([crossdomain.MultiCoil(kspace_model, coil_dim=1)] * self.num_iter)

        self.dc_weight = torch.nn.Parameter(torch.ones(1))

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
        kspace = y.clone()
        zero = torch.zeros(1, 1, 1, 1, 1).to(kspace)

        for idx in range(self.num_iter):
            soft_dc = torch.where(mask.bool(), kspace - y, zero) * self.dc_weight

            kspace = self.kspace_model_list[idx](kspace)
            if kspace.shape[-1] != 2:
                kspace = kspace.permute(0, 1, 3, 4, 2).to(target)
                # this is necessary, but why?
                kspace = torch.view_as_real(kspace[..., 0] + 1j * kspace[..., 1])

            image = utils.complex_mul(
                fft.ifft2(
                    kspace,
                    centered=self.fft_centered,
                    normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                ),
                utils.complex_conj(sensitivity_maps),
            ).sum(self.coil_dim)
            image = self.image_model_list[idx](image.unsqueeze(self.coil_dim)).squeeze(self.coil_dim)

            if not self.no_dc:
                image = fft.fft2(
                    utils.complex_mul(image.unsqueeze(self.coil_dim), sensitivity_maps),
                    centered=self.fft_centered,
                    normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                ).type(image.type())
                image = kspace - soft_dc - image
                image = utils.complex_mul(
                    fft.ifft2(
                        image,
                        centered=self.fft_centered,
                        normalization=self.fft_normalization,
                        spatial_dims=self.spatial_dims,
                    ),
                    utils.complex_conj(sensitivity_maps),
                ).sum(self.coil_dim)

            if idx < self.num_iter - 1:
                kspace = fft.fft2(
                    utils.complex_mul(image.unsqueeze(self.coil_dim), sensitivity_maps),
                    centered=self.fft_centered,
                    normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                ).type(image.type())

        image = torch.view_as_complex(image)
        if target.shape[-1] == 2:
            target = torch.view_as_complex(target)
        _, image = utils.center_crop_to_smallest(target, image)
        return image
