# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from typing import Optional

import torch.nn as nn
from torch import Tensor

import mridc.collections.reconstruction.nn.rim.conv_layers as conv_layers
import mridc.collections.reconstruction.nn.rim.rnn_cells as rnn_cells


class GRUConv2d(nn.Module):
    """
    Implementation of a GRU followed by a number of 2D convolutions inspired by [1].

    References
    ----------
    .. [1] C. Qin, J. Schlemper, J. Caballero, A. N. Price, J. V. Hajnal and D. Rueckert, "Convolutional Recurrent
        Neural Networks for Dynamic MR Image Reconstruction," in IEEE Transactions on Medical Imaging, vol. 38, no. 1,
        pp. 280-290, Jan. 2019, doi: 10.1109/TMI.2018.2863670.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    hidden_channels : int
        Number of hidden channels.
    n_convs : int, optional
        Number of convolutional layers. Default is ``3``.
    activation : torch.nn.Module, optional
        Activation function. Default is ``nn.ReLU()``.
    batchnorm : bool, optional
        If True a batch normalization layer is applied after every convolution. Default is ``False``.
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
        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append(
            rnn_cells.ConvGRUCell(
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
                conv_layers.ConvNonlinear(
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
                conv_layers.ConvNonlinear(
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
        Performs the forward pass of GRUConv2d.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        hx : torch.Tensor, optional
            Hidden state. Default is ``None``.

        Returns
        -------
        torch.Tensor
            Convoluted output.
        """
        if hx is None:
            hx = x.new_zeros((x.size(0), self.hidden_channels, *x.size()[2:]))

        for i, layer in enumerate(self.layers):
            x = layer(x, hx) if i == 0 else layer(x)
        return x
