# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from typing import Any, Optional, Tuple, Union

import torch

import mridc.collections.common.parts.fft as fft
import mridc.collections.common.parts.utils as utils
import mridc.collections.reconstruction.nn.rim.conv_layers as conv_layers
import mridc.collections.reconstruction.nn.rim.rim_utils as rim_utils
import mridc.collections.reconstruction.nn.rim.rnn_cells as rnn_cells


class RIMBlock(torch.nn.Module):
    """
    RIMBlock is a block of Recurrent Inference Machines (RIMs). As presented in [1].

    References
    ----------
    .. [1] Lønning K, Putzky P, Sonke JJ, Reneman L, Caan MW, Welling M. Recurrent inference machines for
        reconstructing heterogeneous MRI data. Medical image analysis. 2019 Apr 1;53:64-78.

    Parameters
    ----------
    recurrent_layer : torch.nn.Module
        Type of the recurrent layer. It can be ``GRU``, ``MGU``, ``IndRNN``. Check ``rnn_cells`` for more dpredictionils.
    conv_filters : list of int
        Number of filters in the convolutional layers.
    conv_kernels : list of int
        Kernel size of the convolutional layers.
    conv_dilations : list of int
        Dilation of the convolutional layers.
    conv_bias : list of bool
        Bias of the convolutional layers.
    recurrent_filters : list of int
        Number of filters in the recurrent layers.
    recurrent_kernels : list of int
        Kernel size of the recurrent layers.
    recurrent_dilations : list of int
        Dilation of the recurrent layers.
    recurrent_bias : list of bool
        Bias of the recurrent layers.
    depth : int
        Number of sequence of convolutional and recurrent layers. Default is ``2``.
    time_steps : int
        Number of reccurent time steps. Default is ``8``.
    conv_dim : int
        Dimension of the convolutional layers. Default is ``2``.
    no_dc : bool
        If ``True`` the DC component is not used. Default is ``True``.
    fft_centered : bool
        If ``True`` the FFT is centered. Default is ``False``.
    fft_normalization : str
        Normalization of the FFT. Default is ``"backward"``.
    spatial_dims : tuple of int
        Spatial dimensions of the input. Default is ``None``.
    coil_dim : int
        Coil dimension of the input. Default is ``1``.
    dimensionality : int
        Dimensionality of the input. Default is ``2``.
    consecutive_slices : int
        Number of consecutive slices. Default is ``1``.
    """

    def __init__(
        self,
        recurrent_layer=None,
        conv_filters=None,
        conv_kernels=None,
        conv_dilations=None,
        conv_bias=None,
        recurrent_filters=None,
        recurrent_kernels=None,
        recurrent_dilations=None,
        recurrent_bias=None,
        depth: int = 2,
        time_steps: int = 8,
        conv_dim: int = 2,
        no_dc: bool = True,
        fft_centered: bool = False,
        fft_normalization: str = "backward",
        spatial_dims: Optional[Tuple[int, int]] = None,
        coil_dim: int = 1,
        dimensionality: int = 2,
        consecutive_slices: int = 1,
    ):
        super(RIMBlock, self).__init__()

        self.input_size = depth * 2
        self.time_steps = time_steps

        self.layers = torch.nn.ModuleList()
        for (
            (conv_features, conv_k_size, conv_dilation, l_conv_bias, nonlinear),
            (rnn_features, rnn_k_size, rnn_dilation, rnn_bias, rnn_type),
        ) in zip(
            zip(conv_filters, conv_kernels, conv_dilations, conv_bias, ["relu", "relu", None]),
            zip(
                recurrent_filters,
                recurrent_kernels,
                recurrent_dilations,
                recurrent_bias,
                [recurrent_layer, recurrent_layer, None],
            ),
        ):
            conv_layer = None

            if conv_features != 0:
                conv_layer = conv_layers.ConvNonlinear(
                    self.input_size,
                    conv_features,
                    conv_dim=conv_dim,
                    kernel_size=conv_k_size,
                    dilation=conv_dilation,
                    bias=l_conv_bias,
                    nonlinear=nonlinear,  # type: ignore
                )
                self.input_size = conv_features

            if rnn_features != 0 and rnn_type is not None:
                if rnn_type.upper() == "GRU":
                    rnn_type = rnn_cells.ConvGRUCell
                elif rnn_type.upper() == "MGU":
                    rnn_type = rnn_cells.ConvMGUCell
                elif rnn_type.upper() == "INDRNN":
                    rnn_type = rnn_cells.IndRNNCell
                else:
                    raise ValueError("Please specify a proper recurrent layer type.")

                rnn_layer = rnn_type(
                    self.input_size,
                    rnn_features,
                    conv_dim=conv_dim,
                    kernel_size=rnn_k_size,
                    dilation=rnn_dilation,
                    bias=rnn_bias,
                )

                self.input_size = rnn_features

                self.layers.append(conv_layers.ConvRNNStack(conv_layer, rnn_layer))

        self.final_layer = torch.nn.Sequential(conv_layer)

        self.recurrent_filters = recurrent_filters

        self.fft_centered = fft_centered
        self.fft_normalization = fft_normalization
        self.spatial_dims = spatial_dims if spatial_dims is not None else [-2, -1]
        self.coil_dim = coil_dim

        self.no_dc = no_dc

        if not self.no_dc:
            self.dc_weight = torch.nn.Parameter(torch.ones(1))
            self.zero = torch.zeros(1, 1, 1, 1, 1)

        self.dimensionality = dimensionality
        self.consecutive_slices = consecutive_slices

    def forward(
        self,
        pred: torch.Tensor,
        masked_kspace: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
        prediction: torch.Tensor = None,
        hx: torch.Tensor = None,
        sigma: float = 1.0,
        keep_prediction: bool = False,
    ) -> Tuple[Any, Union[list, torch.Tensor, None]]:
        """
        Forward pass of the RIMBlock.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted k-space. Shape: ``[batch, coils, height, width, 2]``.
        masked_kspace : torch.Tensor
            Subsampled k-space. Shape: ``[batch, coils, height, width, 2]``.
        sensitivity_maps : torch.Tensor
            Coil sensitivity maps. Shape: ``[batch, coils, height, width, 2]``.
        mask : torch.Tensor
            Subsampling mask. Shape: ``[batch, coils, height, width, 2]``.
        prediction : torch.Tensor, optional
            Initial (zero-filled) prediction. Shape: ``[batch, coils, height, width, 2]``.
        hx : torch.Tensor, optional
            Initial prediction for the hidden state. Shape: ``[batch, coils, height, width, 2]``.
        sigma : float, optional
            Noise level. Default is ``1.0``.
        keep_prediction : bool, optional
            Whether to keep the prediction. Default is ``False``.

        Returns
        -------
        Tuple[Any, Union[list, torch.Tensor, None]]
            Reconstructed image and hidden states.
        """
        if self.dimensionality == 3 or self.consecutive_slices > 1:
            # 2D pred.shape = [batch, coils, height, width, 2]
            # 3D pred.shape = [batch, slices, coils, height, width, 2] -> [batch * slices, coils, height, width, 2]
            batch, slices = masked_kspace.shape[0], masked_kspace.shape[1]
            if isinstance(pred, (tuple, list)):
                pred = pred[-1].detach()
            else:
                pred = pred.reshape([pred.shape[0] * pred.shape[1], *pred.shape[2:]])
            masked_kspace = masked_kspace.reshape(
                [masked_kspace.shape[0] * masked_kspace.shape[1], *masked_kspace.shape[2:]]
            )
            mask = mask.reshape([mask.shape[0] * mask.shape[1], *mask.shape[2:]])
            sensitivity_maps = sensitivity_maps.reshape(
                [sensitivity_maps.shape[0] * sensitivity_maps.shape[1], *sensitivity_maps.shape[2:]]
            )
        else:
            batch = masked_kspace.shape[0]
            slices = 1

            if isinstance(pred, list):
                pred = pred[-1].detach()

        if hx is None:
            hx = [
                masked_kspace.new_zeros((masked_kspace.size(0), f, *masked_kspace.size()[2:-1]))
                for f in self.recurrent_filters
                if f != 0
            ]

        if prediction is None or prediction.ndim < 3:
            prediction = (
                pred
                if keep_prediction
                else torch.sum(
                    utils.complex_mul(
                        fft.ifft2(
                            pred,
                            centered=self.fft_centered,
                            normalization=self.fft_normalization,
                            spatial_dims=self.spatial_dims,
                        ),
                        utils.complex_conj(sensitivity_maps),
                    ),
                    self.coil_dim,
                )
            )

        if (self.consecutive_slices > 1 or self.dimensionality == 3) and prediction.dim() == 5:
            prediction = prediction.reshape([prediction.shape[0] * prediction.shape[1], *prediction.shape[2:]])

        predictions = []
        for _ in range(self.time_steps):
            grad_prediction = rim_utils.log_likelihood_gradient(
                prediction,
                masked_kspace,
                sensitivity_maps,
                mask,
                sigma=sigma,
                fft_centered=self.fft_centered,
                fft_normalization=self.fft_normalization,
                spatial_dims=self.spatial_dims,
                coil_dim=self.coil_dim,
            ).contiguous()

            if self.consecutive_slices > 1 or self.dimensionality == 3:
                grad_prediction = grad_prediction.view(
                    [batch * slices, 4, grad_prediction.shape[2], grad_prediction.shape[3]]
                ).permute(1, 0, 2, 3)

            for h, convrnn in enumerate(self.layers):
                hx[h] = convrnn(grad_prediction, hx[h])
                if self.consecutive_slices > 1 or self.dimensionality == 3:
                    hx[h] = hx[h].squeeze(0)
                grad_prediction = hx[h]

            grad_prediction = self.final_layer(grad_prediction)

            if self.dimensionality == 2:
                grad_prediction = grad_prediction.permute(0, 2, 3, 1)
            elif self.dimensionality == 3:
                grad_prediction = grad_prediction.permute(1, 2, 3, 0)
                for h in range(len(hx)):
                    hx[h] = hx[h].permute(1, 0, 2, 3)

            prediction = prediction + grad_prediction
            predictions.append(prediction)

        prediction = predictions

        if self.no_dc:
            return prediction, hx

        soft_dc = torch.where(mask, pred - masked_kspace, self.zero.to(masked_kspace)) * self.dc_weight
        current_kspace = [
            masked_kspace
            - soft_dc
            - fft.fft2(
                utils.complex_mul(e.unsqueeze(self.coil_dim), sensitivity_maps),
                centered=self.fft_centered,
                normalization=self.fft_normalization,
                spatial_dims=self.spatial_dims,
            )
            for e in prediction
        ]

        return current_kspace, hx
