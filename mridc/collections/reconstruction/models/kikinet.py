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
import mridc.collections.reconstruction.models.crossdomain.multicoil as crossdomain
import mridc.collections.reconstruction.models.didn.didn as didn_
import mridc.collections.reconstruction.models.mwcnn.mwcnn as mwcnn_
import mridc.collections.reconstruction.models.unet_base.unet_block as unet_block
import mridc.core.classes.common as common_classes

__all__ = ["KIKINet"]


class KIKINet(base_models.BaseMRIReconstructionModel, ABC):
    """
    Based on KIKINet implementation [1]. Modified to work with multi-coil k-space data, as presented in Eo, Taejoon, \
    et al.

    References
    ----------

    ..

        Eo, Taejoon, et al. “KIKI-Net: Cross-Domain Convolutional Neural Networks for Reconstructing Undersampled \
        Magnetic Resonance Images.” Magnetic Resonance in Medicine, vol. 80, no. 5, Nov. 2018, pp. 2188–201. PubMed, \
        https://doi.org/10.1002/mrm.27201.

    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # init superclass
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.num_iter = cfg_dict.get("num_iter")
        self.no_dc = cfg_dict.get("no_dc")

        kspace_model_architecture = cfg_dict.get("kspace_model_architecture")

        if kspace_model_architecture == "CONV":
            kspace_model = conv2d.Conv2d(
                in_channels=2,
                out_channels=2,
                hidden_channels=cfg_dict.get("kspace_conv_hidden_channels"),
                n_convs=cfg_dict.get("kspace_conv_n_convs"),
                batchnorm=cfg_dict.get("kspace_conv_batchnorm"),
            )
        elif kspace_model_architecture == "DIDN":
            kspace_model = didn_.DIDN(
                in_channels=2,
                out_channels=2,
                hidden_channels=cfg_dict.get("kspace_didn_hidden_channels"),
                num_dubs=cfg_dict.get("kspace_didn_num_dubs"),
                num_convs_recon=cfg_dict.get("kspace_didn_num_convs_recon"),
            )
        elif kspace_model_architecture in ["UNET", "NORMUNET"]:
            kspace_model = unet_block.NormUnet(
                cfg_dict.get("kspace_unet_num_filters"),
                cfg_dict.get("kspace_unet_num_pool_layers"),
                in_chans=2,
                out_chans=2,
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
                input_channels=2,
                first_conv_hidden_channels=cfg_dict.get("image_mwcnn_hidden_channels"),
                num_scales=cfg_dict.get("image_mwcnn_num_scales"),
                bias=cfg_dict.get("image_mwcnn_bias"),
                batchnorm=cfg_dict.get("image_mwcnn_batchnorm"),
            )
        elif image_model_architecture in ["UNET", "NORMUNET"]:
            image_model = unet_block.NormUnet(
                cfg_dict.get("imspace_unet_num_filters"),
                cfg_dict.get("imspace_unet_num_pool_layers"),
                in_chans=2,
                out_chans=2,
                drop_prob=cfg_dict.get("imspace_unet_dropout_probability"),
                padding_size=cfg_dict.get("imspace_unet_padding_size"),
                normalize=cfg_dict.get("imspace_unet_normalize"),
            )
        else:
            raise NotImplementedError(
                "KIKINet is currently implemented only with image_model_architecture == 'MWCNN' or 'UNet'."
                f"Got {image_model_architecture}."
            )

        self.fft_centered = cfg_dict.get("fft_centered")
        self.fft_normalization = cfg_dict.get("fft_normalization")
        self.spatial_dims = cfg_dict.get("spatial_dims")
        self.coil_dim = cfg_dict.get("coil_dim")

        self.image_model_list = torch.nn.ModuleList([image_model] * self.num_iter)
        self.kspace_model_list = torch.nn.ModuleList([crossdomain.MultiCoil(kspace_model, coil_dim=1)] * self.num_iter)

        if cfg_dict.get("train_loss_fn") == "ssim":
            self.train_loss_fn = losses.SSIMLoss()
        elif cfg_dict.get("train_loss_fn") == "l1":
            self.train_loss_fn = L1Loss()
        elif cfg_dict.get("train_loss_fn") == "mse":
            self.train_loss_fn = torch.nn.MSELoss()
        else:
            raise ValueError("Unknown loss function: {}".format(cfg_dict.get("train_loss_fn")))
        if cfg_dict.get("val_loss_fn") == "ssim":
            self.val_loss_fn = losses.SSIMLoss()
        elif cfg_dict.get("val_loss_fn") == "l1":
            self.val_loss_fn = L1Loss()
        elif cfg_dict.get("val_loss_fn") == "mse":
            self.val_loss_fn = torch.nn.MSELoss()
        else:
            raise ValueError("Unknown loss function: {}".format(cfg_dict.get("val_loss_fn")))

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
        _, image = utils.center_crop_to_smallest(target, image)
        return image
