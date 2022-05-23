# coding=utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NKI-AI/direct/blob/main/direct/nn/recurrentvarnet/recurrentvarnet.py
# Copyright (c) DIRECT Contributors

from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mridc.collections.common.parts.fft import fft2c, ifft2c
from mridc.collections.common.parts.utils import complex_conj, complex_mul
from mridc.collections.reconstruction.models.recurrentvarnet.conv2gru import Conv2dGRU


class RecurrentInit(nn.Module):
    """
    Recurrent State Initializer (RSI) module of Recurrent Variational Network as presented in [1].
    The RSI module learns to initialize the recurrent hidden state :math:`h_0`, input of the first
    RecurrentVarNetBlock of the RecurrentVarNet.

    References
    ----------
    .. [1] Yiasemis, George, et al. “Recurrent Variational Network: A Deep Learning Inverse Problem Solver Applied to the Task of Accelerated MRI Reconstruction.” ArXiv:2111.09639 [Physics], Nov. 2021. arXiv.org, http://arxiv.org/abs/2111.09639.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: Tuple[int, ...],
        dilations: Tuple[int, ...],
        depth: int = 2,
        multiscale_depth: int = 1,
    ):
        """
        Inits RecurrentInit.

        Parameters
        ----------
        in_channels: Input channels.
            int
        out_channels: Number of hidden channels of the recurrent unit of RecurrentVarNet Block.
            int
        channels: Channels :math:`n_d` in the convolutional layers of initializer.
            Tuple[int, ...]
        dilations: Dilations :math:`p` of the convolutional layers of the initializer.
            Tuple[int, ...]
        depth: RecurrentVarNet Block number of layers :math:`n_l`.
            int
        multiscale_depth: Number of feature layers to aggregate for the output, if 1, multi-scale context aggregation
        is disabled.
            int
        """
        super().__init__()

        self.conv_blocks = nn.ModuleList()
        self.out_blocks = nn.ModuleList()
        self.depth = depth
        self.multiscale_depth = multiscale_depth
        tch = in_channels
        for (curr_channels, curr_dilations) in zip(channels, dilations):
            block = [
                nn.ReplicationPad2d(curr_dilations),
                nn.Conv2d(tch, curr_channels, 3, padding=0, dilation=curr_dilations),
            ]
            tch = curr_channels
            self.conv_blocks.append(nn.Sequential(*block))
        tch = np.sum(channels[-multiscale_depth:])
        for _ in range(depth):
            block = [nn.Conv2d(tch, out_channels, 1, padding=0)]
            self.out_blocks.append(nn.Sequential(*block))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes initialization for recurrent unit given input `x`.

        Parameters
        ----------
        x: Initialization for RecurrentInit.

        Returns
        -------
        Initial recurrent hidden state from input `x`.
        """
        features = []
        for block in self.conv_blocks:
            x = F.relu(block(x), inplace=True)
            if self.multiscale_depth > 1:
                features.append(x)
        if self.multiscale_depth > 1:
            x = torch.cat(features[-self.multiscale_depth :], dim=1)
        output_list = []
        for block in self.out_blocks:
            y = F.relu(block(x), inplace=True)
            output_list.append(y)
        return torch.stack(output_list, dim=-1)


class RecurrentVarNetBlock(nn.Module):
    """
    Recurrent Variational Network Block :math:`\mathcal{H}_{\theta_{t}}` as presented in [1].

    References
    ----------
    .. [1] Yiasemis, George, et al. “Recurrent Variational Network: A Deep Learning Inverse Problem Solver Applied to the Task of Accelerated MRI Reconstruction.” ArXiv:2111.09639 [Physics], Nov. 2021. arXiv.org, http://arxiv.org/abs/2111.09639.
    """

    def __init__(
        self,
        in_channels: int = 2,
        hidden_channels: int = 64,
        num_layers: int = 4,
        fft_type: str = "orthogonal",
    ):
        """
        Inits RecurrentVarNetBlock.

        Parameters
        ----------
        in_channels: Input channel number.
            int, Default is 2 for complex data.
        hidden_channels: Hidden channels.
            int, Default: 64.
        num_layers: Number of layers of :math:`n_l` recurrent unit.
            int, Default: 4.
        fft_type: FFT type.
            str, Default: "orthogonal".
        """
        super().__init__()
        self.fft_type = fft_type

        self.learning_rate = nn.Parameter(torch.tensor([1.0]))  # :math:`\alpha_t`
        self.regularizer = Conv2dGRU(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            replication_padding=True,
        )  # Recurrent Unit of RecurrentVarNet Block :math:`\mathcal{H}_{\theta_t}`

    def forward(
        self,
        current_kspace: torch.Tensor,
        masked_kspace: torch.Tensor,
        sampling_mask: torch.Tensor,
        sensitivity_map: torch.Tensor,
        hidden_state: Union[None, torch.Tensor],
        coil_dim: int = 1,
        complex_dim: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes forward pass of RecurrentVarNetBlock.

        Parameters
        ----------
        current_kspace: Current k-space prediction.
            torch.Tensor, shape [batch_size, n_coil, height, width, 2]
        masked_kspace: Subsampled k-space.
            torch.Tensor, shape [batch_size, n_coil, height, width, 2]
        sampling_mask: Sampling mask.
            torch.Tensor, shape [batch_size, 1, height, width, 1]
        sensitivity_map: Coil sensitivities.
            torch.Tensor, shape [batch_size, n_coil, height, width, 2]
        hidden_state: ConvGRU hidden state.
            None or torch.Tensor, shape [batch_size, n_l, height, width, hidden_channels]
        coil_dim: Coil dimension.
            int, Default: 1.
        complex_dim: Complex dimension.
            int, Default: -1.

        Returns
        -------
        new_kspace: New k-space prediction.
            torch.Tensor, shape [batch_size, n_coil, height, width, 2]
        hidden_state: Next hidden state.
            list of torch.Tensor, shape [batch_size, hidden_channels, height, width, num_layers]
        """
        kspace_error = torch.where(
            sampling_mask == 0,
            torch.tensor([0.0], dtype=masked_kspace.dtype).to(masked_kspace.device),
            current_kspace - masked_kspace,
        )

        recurrent_term = torch.cat(
            [
                complex_mul(ifft2c(kspace, fft_type=self.fft_type), complex_conj(sensitivity_map)).sum(coil_dim)
                for kspace in torch.split(current_kspace, 2, complex_dim)
            ],
            dim=complex_dim,
        ).permute(0, 3, 1, 2)

        recurrent_term, hidden_state = self.regularizer(recurrent_term, hidden_state)  # :math:`w_t`, :math:`h_{t+1}`
        recurrent_term = recurrent_term.permute(0, 2, 3, 1)

        recurrent_term = torch.cat(
            [
                fft2c(complex_mul(image.unsqueeze(coil_dim), sensitivity_map), fft_type=self.fft_type)
                for image in torch.split(recurrent_term, 2, complex_dim)
            ],
            dim=complex_dim,
        )

        new_kspace = current_kspace - self.learning_rate * kspace_error + recurrent_term

        return new_kspace, hidden_state
