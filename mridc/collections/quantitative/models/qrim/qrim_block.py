# coding=utf-8
__author__ = "Dimitrios Karkalousos, Chaoping Zhang"

from typing import Any, List, Optional, Tuple, Union

import torch

from mridc.collections.quantitative.models.qrim.utils import SignalForwardModel, analytical_log_likelihood_gradient
from mridc.collections.reconstruction.models.rim.conv_layers import ConvNonlinear, ConvRNNStack
from mridc.collections.reconstruction.models.rim.rnn_cells import ConvGRUCell, ConvMGUCell, IndRNNCell


class qRIMBlock(torch.nn.Module):
    """qRIMBlock extends a block of Recurrent Inference Machines (RIMs)."""

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
        linear_forward_model=None,
        fft_centered: bool = True,
        fft_normalization: str = "ortho",
        spatial_dims: Optional[Tuple[int, int]] = None,
        coil_dim: int = 1,
        coil_combination_method: str = "SENSE",
        dimensionality: int = 2,
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
        fft_centered: If True, the FFT is centered.
        fft_normalization: Normalization of the FFT.
        spatial_dims: Spatial dimensions of the input.
        coil_dim: Coils dimension of the input.
        coil_combination_method: Method to combine the coils.
        dimensionality: Dimensionality of the input.
        """
        super(qRIMBlock, self).__init__()

        self.linear_forward_model = (
            SignalForwardModel(sequence="MEGRE") if linear_forward_model is None else linear_forward_model
        )

        self.input_size = depth * 4
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
                    conv_dim=conv_dim,
                    kernel_size=rnn_k_size,
                    dilation=rnn_dilation,
                    bias=rnn_bias,
                )

                self.input_size = rnn_features

                self.layers.append(ConvRNNStack(conv_layer, rnn_layer))

        self.final_layer = torch.nn.Sequential(conv_layer)

        self.recurrent_filters = recurrent_filters

        self.fft_centered = fft_centered
        self.fft_normalization = fft_normalization
        self.spatial_dims = spatial_dims if spatial_dims is not None else [-2, -1]
        self.coil_dim = coil_dim
        self.coil_combination_method = coil_combination_method

    def forward(
        self,
        pred: torch.Tensor,
        masked_kspace: torch.Tensor,
        R2star_map_init: torch.Tensor,
        S0_map_init: torch.Tensor,
        B0_map_init: torch.Tensor,
        phi_map_init: torch.Tensor,
        TEs: List,
        sensitivity_maps: torch.Tensor,
        sampling_mask: torch.Tensor,
        eta: torch.Tensor = None,
        hx: torch.Tensor = None,
        gamma: torch.Tensor = None,
        keep_eta: bool = False,
    ) -> Tuple[Any, Union[list, torch.Tensor, None]]:
        """
        Forward pass of the RIMBlock.

        Parameters
        ----------
        pred: Initial prediction of the subsampled k-space.
            torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
        masked_kspace: Data.
            torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
        R2star_map_init: Initial R2* map.
            torch.Tensor, shape [batch_size, n_echoes, n_coils, n_x, n_y]
        S0_map_init: Initial S0 map.
            torch.Tensor, shape [batch_size, n_echoes, n_coils, n_x, n_y]
        B0_map_init: Initial B0 map.
            torch.Tensor, shape [batch_size, n_echoes, n_coils, n_x, n_y]
        phi_map_init: Initial phi map.
            torch.Tensor, shape [batch_size, n_echoes, n_coils, n_x, n_y]
        TEs: List of echo times.
            List of int, shape [batch_size, n_echoes]
        sensitivity_maps: Coil sensitivity maps.
            torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
        sampling_mask: Mask of the sampling.
            torch.Tensor, shape [batch_size, 1, n_x, n_y, 2]
        eta: Initial zero-filled.
            torch.Tensor, shape [batch_size, n_x, n_y, 2]
        hx: Initial guess for the hidden state.
        gamma: Scaling normalization factor.
        keep_eta: Whether to keep the eta.

        Returns
        -------
        Reconstructed image and hidden states.
        """
        batch_size = masked_kspace.shape[0]

        if isinstance(pred, list):
            pred = pred[-1].detach()

        if eta is None:
            eta = torch.stack([R2star_map_init, S0_map_init, B0_map_init, phi_map_init], dim=1)

        if hx is None:
            hx = [
                eta.new_zeros((eta.size(0), f, *eta.size()[2:])).to(masked_kspace)
                for f in self.recurrent_filters
                if f != 0
            ]

        R2star_map_init = R2star_map_init * gamma[0]  # type: ignore
        S0_map_init = S0_map_init * gamma[1]  # type: ignore
        B0_map_init = B0_map_init * gamma[2]  # type: ignore
        phi_map_init = phi_map_init * gamma[3]  # type: ignore

        etas = []
        for _ in range(self.time_steps):
            grad_eta = torch.zeros_like(eta)
            for idx in range(batch_size):
                idx_grad_eta = analytical_log_likelihood_gradient(
                    self.linear_forward_model,
                    R2star_map_init[idx],
                    S0_map_init[idx],
                    B0_map_init[idx],
                    phi_map_init[idx],
                    TEs,
                    sensitivity_maps[idx],
                    masked_kspace[idx],
                    sampling_mask[idx],
                    fft_centered=self.fft_centered,
                    fft_normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                    coil_dim=self.coil_dim,
                    coil_combination_method=self.coil_combination_method,
                ).contiguous()
                grad_eta[idx] = idx_grad_eta / 100
                grad_eta[grad_eta != grad_eta] = 0.0

            grad_eta = torch.cat([grad_eta, eta], dim=self.coil_dim - 1).to(masked_kspace)

            for h, convrnn in enumerate(self.layers):
                hx[h] = convrnn(grad_eta, hx[h])
                grad_eta = hx[h]

            grad_eta = self.final_layer(grad_eta)
            eta = eta + grad_eta
            eta_tmp = eta[:, 0, :, :]
            eta_tmp[eta_tmp < 0] = 0
            eta[:, 0, :, :] = eta_tmp

            etas.append(eta)

        return etas, None
