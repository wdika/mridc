# coding=utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NKI-AI/direct/blob/main/direct/nn/conv/conv.py
# Copyright (c) DIRECT Contributors

import torch.nn as nn


class Conv2d(nn.Module):
    """
    Implementation of a simple cascade of 2D convolutions.
    If batchnorm is set to True, batch normalization layer is applied after each convolution.
    """

    def __init__(self, in_channels, out_channels, hidden_channels, n_convs=3, activation=nn.PReLU(), batchnorm=False):
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

        self.conv = []
        for idx in range(n_convs):
            self.conv.append(
                nn.Conv2d(
                    in_channels if idx == 0 else hidden_channels,
                    hidden_channels if idx != n_convs - 1 else out_channels,
                    kernel_size=3,
                    padding=1,
                )
            )
            if batchnorm:
                self.conv.append(nn.BatchNorm2d(hidden_channels if idx != n_convs - 1 else out_channels, eps=1e-4))
            if idx != n_convs - 1:
                self.conv.append(activation)
        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):
        """
        Performs the forward pass of Conv2d.

        Parameters
        ----------
        x: Input tensor.

        Returns
        -------
        Convoluted output.
        """
        if x.dim() == 5:
            x = x.squeeze(1)
            if x.shape[-1] == 2:
                x = x.permute(0, 3, 1, 2)
        return self.conv(x)
