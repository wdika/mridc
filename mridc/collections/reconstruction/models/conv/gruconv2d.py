# coding=utf-8
__author__ = "Dimitrios Karkalousos"

import torch.nn as nn
from torch import Tensor, cat
from typing import Optional

from mridc.collections.reconstruction.models.rim.rnn_cells import ConvGRUCell
from mridc.collections.reconstruction.models.rim.conv_layers import ConvNonlinear


class GRUConv2d(nn.Module):
    """
    Implementation of a GRU followed by a number of 2D convolutions inspired by [1]_.

    References
    ----------

    .. [1] C. Qin, J. Schlemper, J. Caballero, A. N. Price, J. V. Hajnal and D. Rueckert,
    "Convolutional Recurrent Neural Networks for Dynamic MR Image Reconstruction," in IEEE Transactions on Medical
    Imaging, vol. 38, no. 1, pp. 280-290, Jan. 2019, doi: 10.1109/TMI.2018.2863670.
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
        """Inits Conv2d.
        Parameters
        ----------
        in_channels: int
            Number of input channels.
        out_channels: int
            Number of output channels.
        hidden_channels: int
            Number of hidden channels.
        n_convs: int
            Number of convolutional layers.
        activation: nn.Module
            Activation function.
        batchnorm: bool
            If True a batch normalization layer is applied after every convolution.
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
        for i in range(n_convs):
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
        x: torch.Tensor
            Input tensor.
        previous_xs: torch.Tensor
            List of previous input tensors.
        hx: torch.Tensor
            Initial hidden state.
        Returns
        -------
        out: torch.Tensor
            Convoluted output.
        """
        if hx is None:
            hx = x.new_zeros((x.size(0), self.hidden_channels, *x.size()[2:]))

        for i, layer in enumerate(self.layers):
            if i == 0:
                # forward_x = cat([layer(xi, hx) for xi in x])
                # backward_x = cat([layer(xi, hx) for xi in x[::-1]][::-1])
                # x = forward_x + backward_x
                x = layer(x, hx)
            else:
                x = layer(x)
        return x
