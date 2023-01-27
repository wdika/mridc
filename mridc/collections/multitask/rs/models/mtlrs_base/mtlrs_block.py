# coding=utf-8
__author__ = "Dimitrios Karkalousos"

import math
from typing import Dict, List, Optional, Tuple, Union

import torch

import mridc.collections.common.parts.fft as fft
import mridc.collections.common.parts.utils as utils
import mridc.collections.reconstruction.models.rim.rim_block as rim_block
import mridc.collections.reconstruction.models.unet_base.unet_block as unet_block
import mridc.collections.segmentation.models.attention_unet_base.attention_unet_block as attention_unet_block
import mridc.collections.segmentation.models.lambda_unet_base.lambda_unet_block as lambda_unet_block
import mridc.collections.segmentation.models.vnet_base.vnet_block as vnet_block
from mridc.collections.reconstruction.models.rim.conv_layers import ConvNonlinear

__all__ = ["MTLRSBlock"]


class MTLRSBlock(torch.nn.Module):
    """
    Implementation of the Joint Reconstruction & Segmentation Cascades of Independently Recurrent Inference Machines,
    as presented in [1].

    References
    ----------
    .. [1] Placeholder.

    Parameters
    ----------
    reconstruction_module_params : Dict
        Parameters for the reconstruction module.
    segmentation_module_params : Dict
        Parameters for the segmentation module.
    input_channels : int
        Number of input channels.
    magnitude_input : bool
        Whether the input is magnitude or complex. Default is ``True``.
    fft_centered : bool
        Whether the FFT is centered. Default is ``False``.
    fft_normalization : str
        Normalization of the FFT. Default is ``"backward"``.
    spatial_dims : Tuple[int, int]
        Spatial dimensions of the input. Default is ``None``.
    coil_dim : int
        Coil dimension of the input. Default is ``1``.
    dimensionality : int
        Dimensionality of the input. Default is ``2``.
    consecutive_slices : int
        Number of consecutive slices to be used. Default is ``1``.
    coil_combination_method : str
        Coil combination method. Default is ``"SENSE"``.
    normalize_segmentation_output : bool
        Whether to normalize the segmentation output. Default is ``True``   .
    """

    def __init__(
        self,
        reconstruction_module_params: Dict,
        segmentation_module_params: Dict,
        input_channels: int,
        magnitude_input: bool = True,
        fft_centered: bool = False,
        fft_normalization: str = "backward",
        spatial_dims: Optional[Tuple[int, int]] = None,
        coil_dim: int = 1,
        dimensionality: int = 2,
        consecutive_slices: int = 1,
        coil_combination_method: str = "SENSE",
        normalize_segmentation_output: bool = True,
    ):
        super().__init__()

        # General parameters
        self.input_channels = input_channels
        self.magnitude_input = magnitude_input
        self.fft_centered = fft_centered
        self.fft_normalization = fft_normalization
        self.spatial_dims = spatial_dims
        self.coil_dim = coil_dim
        self.dimensionality = dimensionality
        if self.dimensionality != 2:
            raise NotImplementedError(f"Currently only 2D is supported for segmentation, got {self.dimensionality}D.")
        self.consecutive_slices = consecutive_slices
        self.coil_combination_method = coil_combination_method

        # Reconstruction module parameters
        self.reconstruction_module_params = reconstruction_module_params
        self.reconstruction_module_recurrent_filters = self.reconstruction_module_params["recurrent_filters"]
        self.reconstruction_module_time_steps = 8 * math.ceil(self.reconstruction_module_params["time_steps"] / 8)
        self.no_dc = self.reconstruction_module_params["no_dc"]
        self.keep_prediction = self.reconstruction_module_params["keep_prediction"]
        self.reconstruction_module_dimensionality = self.reconstruction_module_params["dimensionality"]
        reconstruction_module_consecutive_slices = (
            self.consecutive_slices if self.reconstruction_module_dimensionality == 3 else 1
        )
        self.reconstruction_module = torch.nn.ModuleList(
            [
                rim_block.RIMBlock(
                    recurrent_layer=self.reconstruction_module_params["recurrent_layer"],
                    conv_filters=self.reconstruction_module_params["conv_filters"],
                    conv_kernels=self.reconstruction_module_params["conv_kernels"],
                    conv_dilations=self.reconstruction_module_params["conv_dilations"],
                    conv_bias=self.reconstruction_module_params["conv_bias"],
                    recurrent_filters=self.reconstruction_module_recurrent_filters,
                    recurrent_kernels=self.reconstruction_module_params["recurrent_kernels"],
                    recurrent_dilations=self.reconstruction_module_params["recurrent_dilations"],
                    recurrent_bias=self.reconstruction_module_params["recurrent_bias"],
                    depth=self.reconstruction_module_params["depth"],
                    time_steps=self.reconstruction_module_time_steps,
                    conv_dim=self.reconstruction_module_params["conv_dim"],
                    no_dc=self.no_dc,
                    fft_centered=self.fft_centered,
                    fft_normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                    coil_dim=self.coil_dim - 1,
                    dimensionality=self.reconstruction_module_dimensionality,
                    consecutive_slices=reconstruction_module_consecutive_slices,
                )
                for _ in range(self.reconstruction_module_params["num_cascades"])
            ]
        )
        # Keep estimation through the cascades if keep_prediction is True or re-estimate it if False.
        self.reconstruction_module_keep_prediction = self.reconstruction_module_params["keep_prediction"]
        # initialize weights if not using pretrained cirim
        if not self.reconstruction_module_params["pretrained"]:
            std_init_range = 1 / self.reconstruction_module_recurrent_filters[0] ** 0.5
            self.reconstruction_module.apply(lambda module: utils.rnn_weights_init(module, std_init_range))
        self.dc_weight = torch.nn.Parameter(torch.ones(1))
        self.reconstruction_module_accumulate_predictions = self.reconstruction_module_params["accumulate_predictions"]

        # Segmentation module parameters
        self.segmentation_module_params = segmentation_module_params
        segmentation_module = self.segmentation_module_params["segmentation_module"]
        self.segmentation_module_output_channels = self.segmentation_module_params["output_channels"]
        if segmentation_module.lower() == "unet":
            segmentation_module = unet_block.Unet(
                in_chans=self.input_channels,
                out_chans=self.segmentation_module_output_channels,
                chans=self.segmentation_module_params["channels"],
                num_pool_layers=self.segmentation_module_params["pooling_layers"],
                drop_prob=self.segmentation_module_params["dropout"],
            )
        elif segmentation_module.lower() == "attentionunet":
            segmentation_module = attention_unet_block.AttentionUnet(
                in_chans=self.input_channels,
                out_chans=self.segmentation_module_output_channels,
                chans=self.segmentation_module_params["channels"],
                num_pool_layers=self.segmentation_module_params["pooling_layers"],
                drop_prob=self.segmentation_module_params["dropout"],
            )
        elif segmentation_module.lower() == "lambdaunet":
            segmentation_module = lambda_unet_block.LambdaBlock(
                in_chans=self.input_channels,
                out_chans=self.segmentation_module_output_channels,
                drop_prob=self.segmentation_module_params["dropout"],
                temporal_kernel=self.segmentation_module_params["temporal_kernel"],
                num_slices=self.consecutive_slices,
            )
        elif segmentation_module.lower() == "vnet":
            segmentation_module = vnet_block.VNet(
                in_chans=self.input_channels,
                out_chans=self.segmentation_module_output_channels,
                act=self.segmentation_module_params["activation"],
                drop_prob=self.segmentation_module_params["dropout"],
                bias=self.segmentation_module_params["bias"],
            )
        elif segmentation_module.lower() == "convlayer":
            segmentation_module = torch.nn.Sequential(
                ConvNonlinear(
                    self.input_channels,
                    self.segmentation_module_output_channels,
                    conv_dim=self.segmentation_module_params["conv_dim"],
                    kernel_size=3,
                    dilation=1,
                    bias=False,
                    nonlinear=None,  # No nonlinear activation
                )
            )
        else:
            raise ValueError(f"Segmentation module {segmentation_module} not implemented.")
        self.segmentation_module = segmentation_module

        self.normalize_segmentation_output = normalize_segmentation_output

    def forward(
        self,
        y: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
        init_reconstruction_pred: torch.Tensor,
        target_reconstruction: torch.Tensor,
        hx: torch.Tensor = None,
        sigma: float = 1.0,
    ) -> Tuple[Union[List, torch.Tensor], torch.Tensor]:
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
        hx : torch.Tensor, optional
            Initial hidden state for the RNN. Default is ``None``.
        sigma : float, optional
            Standard deviation of the noise. Default is ``1.0``.

        Returns
        -------
        Tuple[Union[List, torch.Tensor], torch.Tensor]
            Tuple containing the predicted reconstruction and segmentation.
        """
        if self.consecutive_slices > 1 and self.reconstruction_module_dimensionality == 2:
            # Do per slice reconstruction
            pred_reconstruction_slices = []
            for slice_idx in range(self.consecutive_slices):
                y_slice = y[:, slice_idx, ...]
                prediction_slice = y_slice.clone()
                sensitivity_maps_slice = sensitivity_maps[:, slice_idx, ...]
                mask_slice = mask[:, 0, ...]
                init_reconstruction_pred_slice = init_reconstruction_pred[:, slice_idx, ...]
                _pred_reconstruction_slice = (
                    None
                    if init_reconstruction_pred_slice is None or init_reconstruction_pred_slice.dim() < 4
                    else init_reconstruction_pred_slice
                )
                target_reconstruction_slice = target_reconstruction[:, slice_idx, ...]
                cascades_predictions = []
                for i, cascade in enumerate(self.reconstruction_module):
                    # Forward pass through the cascades
                    prediction_slice, hx = cascade(
                        prediction_slice,
                        y_slice,
                        sensitivity_maps_slice,
                        mask_slice,
                        _pred_reconstruction_slice,
                        hx,
                        sigma,
                        keep_prediction=False if i == 0 else self.keep_prediction,
                    )
                    time_steps_predictions = [
                        self.process_intermediate_pred(pred, sensitivity_maps_slice, target_reconstruction_slice)
                        for pred in prediction_slice
                    ]
                    cascades_predictions.append(torch.stack(time_steps_predictions, dim=0))
                pred_reconstruction_slices.append(torch.stack(cascades_predictions, dim=0))
            preds = torch.stack(pred_reconstruction_slices, dim=3)

            cascades_predictions = [
                [
                    preds[cascade_prediction, time_step_prediction, ...]
                    for time_step_prediction in range(preds.shape[1])
                ]
                for cascade_prediction in range(preds.shape[0])
            ]
        else:
            prediction = y.clone()
            _pred_reconstruction = (
                None
                if init_reconstruction_pred is None or init_reconstruction_pred.dim() < 4
                else init_reconstruction_pred
            )
            sigma = 1.0
            cascades_predictions = []
            for i, cascade in enumerate(self.reconstruction_module):
                # Forward pass through the cascades
                prediction, hx = cascade(
                    prediction,
                    y,
                    sensitivity_maps,
                    mask,
                    _pred_reconstruction,
                    hx,
                    sigma,
                    keep_prediction=False if i == 0 else self.keep_prediction,
                )
                time_steps_predictions = [
                    self.process_intermediate_pred(pred, sensitivity_maps, target_reconstruction)
                    for pred in prediction
                ]
                cascades_predictions.append(time_steps_predictions)
        pred_reconstruction = cascades_predictions

        _pred_reconstruction = pred_reconstruction
        if isinstance(_pred_reconstruction, list):
            _pred_reconstruction = _pred_reconstruction[-1]
        if isinstance(_pred_reconstruction, list):
            _pred_reconstruction = _pred_reconstruction[-1]
        if _pred_reconstruction.shape[-1] != 2:  # type: ignore
            _pred_reconstruction = torch.view_as_real(_pred_reconstruction)
        if self.consecutive_slices > 1 and _pred_reconstruction.dim() == 5:
            _pred_reconstruction = _pred_reconstruction.reshape(  # type: ignore
                # type: ignore
                _pred_reconstruction.shape[0] * _pred_reconstruction.shape[1],
                *_pred_reconstruction.shape[2:],  # type: ignore
            )
        if _pred_reconstruction.shape[-1] == 2:  # type: ignore
            if self.input_channels == 1:
                _pred_reconstruction = torch.view_as_complex(_pred_reconstruction).unsqueeze(1)
                if self.magnitude_input:
                    _pred_reconstruction = torch.abs(_pred_reconstruction)
            elif self.input_channels == 2:
                if self.magnitude_input:
                    raise ValueError("Magnitude input is not supported for 2-channel input.")
                _pred_reconstruction = _pred_reconstruction.permute(0, 3, 1, 2)  # type: ignore
            else:
                raise ValueError("The input channels must be either 1 or 2. Found: {}".format(self.input_channels))
        else:
            _pred_reconstruction = _pred_reconstruction.unsqueeze(1)

        with torch.no_grad():
            _pred_reconstruction = torch.nn.functional.group_norm(_pred_reconstruction, num_groups=1)

        pred_segmentation = self.segmentation_module(_pred_reconstruction)

        pred_segmentation = torch.abs(pred_segmentation)

        if self.normalize_segmentation_output:
            pred_segmentation = pred_segmentation / torch.max(pred_segmentation)

        if self.consecutive_slices > 1:
            # get batch size and number of slices from y, because if the reconstruction module is used they will
            # not be saved before
            pred_segmentation = pred_segmentation.view([y.shape[0], y.shape[1], *pred_segmentation.shape[1:]])

        return pred_reconstruction, pred_segmentation, hx  # type: ignore

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
        if not self.no_dc or do_coil_combination:
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
