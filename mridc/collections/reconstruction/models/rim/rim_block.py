# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from typing import Any, Tuple, Union

import torch

from mridc.collections.common.parts.fft import fft2c, ifft2c
from mridc.collections.common.parts.utils import complex_conj, complex_mul
from mridc.collections.reconstruction.models.rim.conv_layers import ConvNonlinear, ConvRNNStack
from mridc.collections.reconstruction.models.rim.rnn_cells import ConvGRUCell, ConvMGUCell, IndRNNCell
from mridc.collections.reconstruction.models.rim.utils import log_likelihood_gradient


class RIMBlock(torch.nn.Module):
    """RIMBlock is a block of Recurrent Inference Machines (RIMs)."""

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
        no_dc: bool = False,
        fft_type: str = "orthogonal",
    ):
        """
        Initialize the RIMBlock.

        Parameters
        ----------
        recurrent_layer: Type of recurrent layer.
        conv_filters: Number of filters in the convolutional layers.
        conv_kernels: Kernel size of the convolutional layers.
        conv_dilations: Dilation of the convolutional layers.
        conv_bias: Bias of the convolutional layers.
        recurrent_filters: Number of filters in the recurrent layers.
        recurrent_kernels: Kernel size of the recurrent layers.
        recurrent_dilations: Dilation of the recurrent layers.
        recurrent_bias: Bias of the recurrent layers.
        depth: Number of layers in the block.
        time_steps: Number of time steps in the block.
        conv_dim: Dimension of the convolutional layers.
        no_dc: If True, the DC component is removed from the input.
        fft_type: Type of FFT.
        """
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
                conv_layer = ConvNonlinear(
                    self.input_size,
                    conv_features,
                    conv_dim=conv_dim,
                    kernel_size=conv_k_size,
                    dilation=conv_dilation,
                    bias=l_conv_bias,
                    nonlinear=nonlinear,
                )
                self.input_size = conv_features

            if rnn_features != 0 and rnn_type is not None:
                if rnn_type.upper() == "GRU":
                    rnn_type = ConvGRUCell
                elif rnn_type.upper() == "MGU":
                    rnn_type = ConvMGUCell
                elif rnn_type.upper() == "INDRNN":
                    rnn_type = IndRNNCell
                else:
                    raise ValueError("Please specify a proper recurrent layer type.")

                rnn_layer = rnn_type(
                    self.input_size,
                    rnn_features,
                    conv_dim=2,
                    kernel_size=rnn_k_size,
                    dilation=rnn_dilation,
                    bias=rnn_bias,
                )

                self.input_size = rnn_features

                self.layers.append(ConvRNNStack(conv_layer, rnn_layer))

        self.final_layer = torch.nn.Sequential(conv_layer)

        self.recurrent_filters = recurrent_filters
        self.fft_type = fft_type

        self.no_dc = no_dc

        if not self.no_dc:
            self.dc_weight = torch.nn.Parameter(torch.ones(1))
            self.zero = torch.zeros(1, 1, 1, 1, 1)

    def forward(
        self,
        pred: torch.Tensor,
        masked_kspace: torch.Tensor,
        sense: torch.Tensor,
        mask: torch.Tensor,
        eta: torch.Tensor = None,
        hx: torch.Tensor = None,
        sigma: float = 1.0,
        keep_eta: bool = False,
    ) -> Tuple[Any, Union[list, torch.Tensor, None]]:
        """
        Forward pass of the RIMBlock.

        Parameters
        ----------
        pred: Predicted k-space.
        masked_kspace: Subsampled k-space.
        sense: Coil sensitivity maps.
        mask: Sample mask.
        eta: Initial guess for the eta.
        hx: Initial guess for the hidden state.
        sigma: Noise level.
        keep_eta: Whether to keep the eta.

        Returns
        -------
        Reconstructed image and hidden states.
        """
        if hx is None:
            hx = [
                masked_kspace.new_zeros((masked_kspace.size(0), f, *masked_kspace.size()[2:-1]))
                for f in self.recurrent_filters
                if f != 0
            ]

        if isinstance(pred, list):
            pred = pred[-1].detach()

        if eta is None or eta.ndim < 3:
            eta = (
                pred
                if keep_eta
                else torch.sum(
                    complex_mul(ifft2c(pred, fft_type=self.fft_type), complex_conj(sense)),
                    1,
                )
            )

        etas = []
        for _ in range(self.time_steps):
            grad_eta = log_likelihood_gradient(
                eta, masked_kspace, sense, mask, sigma=sigma, fft_type=self.fft_type
            ).contiguous()

            for h, convrnn in enumerate(self.layers):
                hx[h] = convrnn(grad_eta, hx[h])
                grad_eta = hx[h]

            eta = eta + self.final_layer(grad_eta).permute(0, 2, 3, 1)
            etas.append(eta)

        eta = etas

        if self.no_dc:
            return eta, None

        soft_dc = torch.where(mask, pred - masked_kspace, self.zero.to(masked_kspace)) * self.dc_weight
        current_kspace = [
            masked_kspace - soft_dc - fft2c(complex_mul(e.unsqueeze(1), sense), fft_type=self.fft_type) for e in eta
        ]

        return current_kspace, None
