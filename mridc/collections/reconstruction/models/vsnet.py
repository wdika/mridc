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
import mridc.collections.reconstruction.models.mwcnn.mwcnn as mwcnn_
import mridc.collections.reconstruction.models.unet_base.unet_block as unet_block
import mridc.collections.reconstruction.models.variablesplittingnet.vsnet_block as vsnet_block
import mridc.core.classes.common as common_classes

__all__ = ["VSNet"]


class VSNet(base_models.BaseMRIReconstructionModel, ABC):
    """
    Implementation of the Variable-Splitting Net, as presented in Duan, J. et al.

    References
    ----------

    ..

        Duan, J. et al. (2019) ‘Vs-net: Variable splitting network for accelerated parallel MRI reconstruction’, \
        Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture \
        Notes in Bioinformatics), 11767 LNCS, pp. 713–722. doi: 10.1007/978-3-030-32251-9_78.

    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # init superclass
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        num_cascades = cfg_dict.get("num_cascades")
        self.fft_normalization = cfg_dict.get("fft_normalization")
        self.spatial_dims = cfg_dict.get("spatial_dims")
        self.coil_dim = cfg_dict.get("coil_dim")
        self.num_cascades = cfg_dict.get("num_cascades")

        image_model_architecture = cfg_dict.get("imspace_model_architecture")
        if image_model_architecture == "CONV":
            image_model = conv2d.Conv2d(
                in_channels=2,
                out_channels=2,
                hidden_channels=cfg_dict.get("imspace_conv_hidden_channels"),
                n_convs=cfg_dict.get("imspace_conv_n_convs"),
                batchnorm=cfg_dict.get("imspace_conv_batchnorm"),
            )
        elif image_model_architecture == "MWCNN":
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

        self.coil_combination_method = cfg_dict.get("coil_combination_method")

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
        sensitivity_maps = self.sens_net(y, mask) if self.use_sens_net else sensitivity_maps
        image = self.model(y, sensitivity_maps, mask)
        image = torch.view_as_complex(
            utils.coil_combination(
                fft.ifft2(
                    image,
                    centered=self.fft_centered,
                    normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                ),
                sensitivity_maps,
                method=self.coil_combination_method,
                dim=self.coil_dim,
            )
        )
        _, image = utils.center_crop_to_smallest(target, image)
        return image
