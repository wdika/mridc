# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from abc import ABC

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

import mridc.collections.reconstruction.nn.base as base_models
import mridc.collections.reconstruction.nn.mwcnn.mwcnn as mwcnn_
import mridc.core.classes.common as common_classes
from mridc.collections.common.parts import fft, utils
from mridc.collections.reconstruction.nn.conv import conv2d
from mridc.collections.reconstruction.nn.unet_base import unet_block
from mridc.collections.reconstruction.nn.variablesplittingnet import vsnet_block

__all__ = ["VSNet"]


class VSNet(base_models.BaseMRIReconstructionModel, ABC):  # type: ignore
    """
    Implementation of the Variable-Splitting Net, as presented in [1].

    References
    ----------
    .. [1] Duan, J. et al. (2019) ‘Vs-net: Variable splitting network for accelerated parallel MRI reconstruction’,
        Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture
        Notes in Bioinformatics), 11767 LNCS, pp. 713–722. doi: 10.1007/978-3-030-32251-9_78.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        num_cascades = cfg_dict.get("num_cascades")
        self.num_cascades = cfg_dict.get("num_cascades")

        image_model_architecture = cfg_dict.get("imspace_model_architecture")
        if image_model_architecture == "CONV":
            image_model = conv2d.Conv2d(
                in_channels=cfg_dict.get("imspace_in_channels", 2),
                out_channels=cfg_dict.get("imspace_out_channels", 2),
                hidden_channels=cfg_dict.get("imspace_conv_hidden_channels"),
                n_convs=cfg_dict.get("imspace_conv_n_convs"),
                batchnorm=cfg_dict.get("imspace_conv_batchnorm"),
            )
        elif image_model_architecture == "MWCNN":
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
                "VSNet is currently implemented only with image_model_architecture == 'MWCNN' or 'UNet'."
                f"Got {image_model_architecture}."
            )

        image_model = torch.nn.ModuleList([image_model] * num_cascades)
        data_consistency_model = torch.nn.ModuleList([vsnet_block.DataConsistencyLayer()] * num_cascades)
        weighted_average_model = torch.nn.ModuleList([vsnet_block.WeightedAverageTerm()] * num_cascades)

        self.model = vsnet_block.VSNetBlock(
            denoiser_block=image_model,
            data_consistency_block=data_consistency_model,
            weighted_average_block=weighted_average_model,
            num_cascades=num_cascades,
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
        sensitivity_maps = self.sens_net(y, mask) if self.use_sens_net else sensitivity_maps
        prediction = self.model(y, sensitivity_maps, mask)
        prediction = torch.view_as_complex(
            utils.coil_combination_method(
                fft.ifft2(
                    prediction,
                    centered=self.fft_centered,
                    normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                ),
                sensitivity_maps,
                method=self.coil_combination_method,
                dim=self.coil_dim,
            )
        )
        if target.shape[-1] == 2:
            target = torch.view_as_complex(target)
        _, prediction = utils.center_crop_to_smallest(target, prediction)
        return prediction
