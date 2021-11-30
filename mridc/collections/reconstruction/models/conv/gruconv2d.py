# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from typing import Optional

import torch.nn as nn
from torch import Tensor

from mridc.collections.reconstruction.models.rim.conv_layers import ConvNonlinear
from mridc.collections.reconstruction.models.rim.rnn_cells import ConvGRUCell


class GRUConv2d(nn.Module):
    """
    Implementation of a GRU followed by a number of 2D convolutions inspired by [1]_.

    References
    ----------
    .. [1] C. Qin, J. Schlemper, J. Caballero, A. N. Price, J. V. Hajnal and D. Rueckert, "Convolutional Recurrent Neural Networks for Dynamic MR Image Reconstruction," in IEEE Transactions on Medical Imaging, vol. 38, no. 1, pp. 280-290, Jan. 2019, doi: 10.1109/TMI.2018.2863670.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        n_convs=3,
        activation="ReLU",
        batchnorm=False,
    ):
        """
        Inits Conv2d.

        Parameters
        ----------
        in_channels: Number of input channels.
            int
        out_channels: Number of output channels.
            int
        hidden_channels: Number of hidden channels.
            int
        n_convs: Number of convolutional layers.
            int
        activation: Activation function.
            torch.nn.Module
        batchnorm: If True a batch normalization layer is applied after every convolution.
            bool
        """
        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append(
            ConvGRUCell(
                in_channels,
                hidden_channels,
                conv_dim=2,
                kernel_size=3,
                dilation=1,
                bias=False,
            )
        )
        for _ in range(n_convs):
            self.layers.append(
                ConvNonlinear(
                    hidden_channels,
                    hidden_channels,
                    conv_dim=2,
                    kernel_size=3,
                    dilation=1,
                    bias=False,
                    nonlinear=activation,
                )
            )
        self.layers.append(
            nn.Sequential(
                ConvNonlinear(
                    hidden_channels,
                    out_channels,
                    conv_dim=2,
                    kernel_size=3,
                    dilation=1,
                    bias=False,
                    nonlinear=activation,
                )
            )
        )

        self.hidden_channels = hidden_channels

    def forward(self, x, hx: Optional[Tensor] = None):
        """
        Performs the forward pass of Conv2d.

        Parameters
        ----------
        x: Input tensor.
            torch.Tensor
        hx: Initial hidden state.
            torch.Tensor

        Returns
        -------
        Convoluted output.
        """
        if hx is None:
            hx = x.new_zeros((x.size(0), self.hidden_channels, *x.size()[2:]))

        for i, layer in enumerate(self.layers):
            x = layer(x, hx) if i == 0 else layer(x)
        return x
