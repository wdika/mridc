# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from abc import ABC
from typing import Any, Dict, Tuple, Union

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch import nn

import mridc.collections.multitask.rs.nn.base as base_rs_models
import mridc.core.classes.common as common_classes
from mridc.collections.common.parts import fft, utils
from mridc.collections.multitask.rs.nn.idslr_base import idslr_block
from mridc.collections.reconstruction.nn.rim import conv_layers

__all__ = ["SegNet"]


class SegNet(base_rs_models.BaseMRIReconstructionSegmentationModel, ABC):  # type: ignore
    """
    Implementation of the Segmentation Network MRI, as described in, as presented in [1].

    References
    ----------
    .. [1] Sun, L., Fan, Z., Ding, X., Huang, Y., Paisley, J. (2019). Joint CS-MRI Reconstruction and Segmentation with
        a Unified Deep Network. In: Chung, A., Gee, J., Yushkevich, P., Bao, S. (eds) Information Processing in Medical
         Imaging. IPMI 2019. Lecture Notes in Computer Science(), vol 11492. Springer, Cham.
         https://doi.org/10.1007/978-3-030-20351-1_38
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.use_reconstruction_module = cfg_dict.get("use_reconstruction_module", True)

        self.dimensionality = cfg_dict.get("dimensionality", 2)
        if self.dimensionality != 2:
            raise NotImplementedError(f"Currently only 2D is supported for segmentation, got {self.dimensionality}D.")

        self.input_channels = cfg_dict.get("input_channels", 2)
        reconstruction_out_chans = cfg_dict.get("reconstruction_module_output_channels", 2)
        segmentation_out_chans = cfg_dict.get("segmentation_module_output_channels", 1)
        chans = cfg_dict.get("channels", 32)
        num_pools = cfg_dict.get("num_pools", 4)
        drop_prob = cfg_dict.get("drop_prob", 0.0)
        normalize = cfg_dict.get("normalize", False)
        padding = cfg_dict.get("padding", False)
        padding_size = cfg_dict.get("padding_size", 11)
        self.norm_groups = cfg_dict.get("norm_groups", 2)
        num_cascades = cfg_dict.get("num_cascades", 5)

        self.reconstruction_encoder = nn.ModuleList(
            [
                idslr_block.UnetEncoder(
                    chans=chans,
                    num_pools=num_pools,
                    in_chans=self.input_channels,
                    drop_prob=drop_prob,
                    normalize=normalize,
                    padding=padding,
                    padding_size=padding_size,
                    norm_groups=self.norm_groups,
                )
                for _ in range(num_cascades)
            ]
        )
        self.reconstruction_decoder = nn.ModuleList(
            [
                idslr_block.UnetDecoder(
                    chans=chans,
                    num_pools=num_pools,
                    out_chans=reconstruction_out_chans,
                    drop_prob=drop_prob,
                    normalize=normalize,
                    padding=padding,
                    padding_size=padding_size,
                    norm_groups=self.norm_groups,
                )
                for _ in range(num_cascades)
            ]
        )
        self.segmentation_decoder = nn.ModuleList(
            [
                idslr_block.UnetDecoder(
                    chans=chans,
                    num_pools=num_pools,
                    out_chans=segmentation_out_chans,
                    drop_prob=drop_prob,
                    normalize=normalize,
                    padding=padding,
                    padding_size=padding_size,
                    norm_groups=self.norm_groups,
                )
                for _ in range(num_cascades)
            ]
        )

        self.segmentation_final_layer = torch.nn.Sequential(
            conv_layers.ConvNonlinear(
                segmentation_out_chans * num_cascades,
                segmentation_out_chans,
                conv_dim=cfg_dict.get("segmentation_final_layer_conv_dim", 2),
                kernel_size=cfg_dict.get("segmentation_final_layer_kernel_size", 3),
                dilation=cfg_dict.get("segmentation_final_layer_dilation", 1),
                bias=cfg_dict.get("segmentation_final_layer_bias", False),
                nonlinear=cfg_dict.get("segmentation_final_layer_nonlinear", "relu"),
            )
        )

        self.magnitude_input = cfg_dict.get("magnitude_input", True)
        self.normalize_segmentation_output = cfg_dict.get("normalize_segmentation_output", True)

        self.dc = idslr_block.DC()

    @common_classes.typecheck()  # type: ignore
    def forward(  # noqa: W0221
        self,
        y: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
        init_reconstruction_pred: torch.Tensor,
        target_reconstruction: torch.Tensor,
    ) -> Tuple[Any, Any]:
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
        init_reconstruction_pred : torch.Tensor
            Initial reconstruction prediction. Shape [batch_size, n_x, n_y, 2]
        target_reconstruction : torch.Tensor
            Target reconstruction. Shape [batch_size, n_x, n_y, 2]

        Returns
        -------
        Tuple[Union[List, torch.Tensor], torch.Tensor]
            Tuple containing the predicted reconstruction and segmentation.
        """
        if self.consecutive_slices > 1:
            batch, slices = y.shape[0], y.shape[1]
            y = y.reshape(y.shape[0] * y.shape[1], *y.shape[2:])  # type: ignore
            sensitivity_maps = sensitivity_maps.reshape(
                # type: ignore
                sensitivity_maps.shape[0] * sensitivity_maps.shape[1],
                *sensitivity_maps.shape[2:],
            )
            mask = mask.reshape(mask.shape[0] * mask.shape[1], *mask.shape[2:])  # type: ignore

        # In case of deviating number of coils, we need to pad up to maximum number of coils == number of input \
        # channels for the reconstruction module
        num_coils = y.shape[1]
        if num_coils * 2 != self.input_channels:
            num_coils_to_add = (self.input_channels - num_coils * 2) // 2
            dummy_coil_data = torch.zeros_like(torch.movedim(y, self.coil_dim, 0)[0]).unsqueeze(self.coil_dim)
            for _ in range(num_coils_to_add):
                y = torch.cat([y, dummy_coil_data], dim=self.coil_dim)
                sensitivity_maps = torch.cat([sensitivity_maps, dummy_coil_data], dim=self.coil_dim)

        y_prediction = y.clone()
        pred_segmentations = []
        for re, rd, sd in zip(self.reconstruction_encoder, self.reconstruction_decoder, self.segmentation_decoder):
            init_reconstruction_pred = fft.ifft2(
                y_prediction, self.fft_centered, self.fft_normalization, self.spatial_dims
            )
            output = re(init_reconstruction_pred)
            reconstruction_encoder_prediction, padding_size = output[0].copy(), output[2]
            with torch.no_grad():
                pred_segmentation_input = [
                    torch.nn.functional.group_norm(x, num_groups=self.norm_groups)
                    for x in reconstruction_encoder_prediction
                ]
                if self.magnitude_input:
                    pred_segmentation_input = [torch.abs(x) for x in pred_segmentation_input]
            pred_segmentations.append(sd(pred_segmentation_input, iscomplex=False, pad_sizes=padding_size))
            reconstruction_decoder_prediction = rd(*output)
            reconstruction_decoder_prediction_kspace = fft.fft2(
                reconstruction_decoder_prediction,
                centered=self.fft_centered,
                normalization=self.fft_normalization,
                spatial_dims=self.spatial_dims,
            )
            y_prediction = self.dc(reconstruction_decoder_prediction_kspace, y, mask)

        pred_reconstruction = self.process_intermediate_pred(
            y_prediction, sensitivity_maps, target_reconstruction, do_coil_combination=True
        )

        pred_segmentation = self.segmentation_final_layer(torch.cat(pred_segmentations, dim=1))
        pred_segmentations.append(pred_segmentation)

        if self.normalize_segmentation_output:
            pred_segmentations = [x / torch.max(x) for x in pred_segmentations]

        if self.consecutive_slices > 1:
            # get batch size and number of slices from y, because if the reconstruction module is used they will not
            # be saved before
            pred_reconstruction = pred_reconstruction.view([batch, slices, *pred_reconstruction.shape[1:]])
            pred_segmentations = [x.view([batch, slices, *x.shape[1:]]) for x in pred_segmentations]

        return pred_reconstruction, pred_segmentations

    def process_segmentation_loss(self, target: torch.Tensor, prediction: torch.Tensor) -> Dict:
        """
        Processes the segmentation loss.

        Parameters
        ----------
        target : torch.Tensor
            Target data of shape [batch_size, nr_classes, n_x, n_y].
        prediction : torch.Tensor
            Prediction of shape [batch_size, nr_classes, n_x, n_y].

        Returns
        -------
        Dict
            Dictionary containing the (multiple) loss values. For example, if the cross entropy loss and the dice loss
            are used, the dictionary will contain the keys ``cross_entropy_loss``, ``dice_loss``, and
            (combined) ``segmentation_loss``.
        """
        loss_dict = {"cross_entropy_loss": 0.0, "dice_loss": 0.0}
        cross_entropy_loss = []  # type: ignore
        dice_loss = []  # type: ignore
        for i in range(len(prediction)):  # noqa: C0200
            if self.segmentation_loss_fn["cross_entropy"] is not None:
                cross_entropy_loss.append(
                    self.segmentation_loss_fn["cross_entropy"].cpu()(
                        target.argmax(1).detach().cpu(), prediction[i].detach().cpu()
                    )
                )
            if self.segmentation_loss_fn["dice"] is not None:
                _, loss_dice = self.segmentation_loss_fn["dice"](target, prediction[i])  # noqa: E1102
                dice_loss.append(loss_dice)
        if self.segmentation_loss_fn["cross_entropy"] is not None:
            loss_dict["cross_entropy_loss"] = (
                torch.stack(cross_entropy_loss).mean() * self.cross_entropy_loss_weighting_factor
            )
        if self.segmentation_loss_fn["dice"] is not None:
            loss_dict["dice_loss"] = torch.stack(dice_loss).mean() * self.dice_loss_weighting_factor
        loss_dict["segmentation_loss"] = loss_dict["cross_entropy_loss"] + loss_dict["dice_loss"]
        return loss_dict

    def process_intermediate_pred(
        self,
        prediction: Union[list, torch.Tensor],
        sensitivity_maps: torch.Tensor,
        target: torch.Tensor,
        do_coil_combination: bool = False,
    ) -> torch.Tensor:
        """
        Processes the intermediate prediction.

        Parameters
        ----------
        prediction : torch.Tensor
            Intermediate prediction. Shape [batch_size, n_coils, n_x, n_y, 2]
        sensitivity_maps : torch.Tensor
            Coil sensitivity maps. Shape [batch_size, n_coils, n_x, n_y, 2]
        target : torch.Tensor
            Target data to crop to size. Shape [batch_size, n_x, n_y, 2]
        do_coil_combination : bool
            Whether to do coil combination. In this case the prediction is in k-space. Default is ``False``.

        Returns
        -------
        torch.Tensor, shape [batch_size, n_x, n_y, 2]
            Processed prediction.
        """
        # Take the last time step of the prediction
        if do_coil_combination:
            prediction = fft.ifft2(
                prediction,
                centered=self.fft_centered,
                normalization=self.fft_normalization,
                spatial_dims=self.spatial_dims,
            )
            prediction = utils.coil_combination_method(
                prediction, sensitivity_maps, method=self.coil_combination_method, dim=self.coil_dim
            )
        prediction = torch.view_as_complex(prediction)
        if target.shape[-1] == 2:
            target = torch.view_as_complex(target)
        _, prediction = utils.center_crop_to_smallest(target, prediction)
        return prediction
