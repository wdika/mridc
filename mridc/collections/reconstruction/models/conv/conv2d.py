# coding=utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NKI-AI/direct/blob/main/direct/nn/conv/conv.py

import torch
import torch.nn as nn


class Conv2d(nn.Module):
    """
    Implementation of a simple cascade of 2D convolutions. If batchnorm is set to True, batch normalization layer is
    applied after each convolution.

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
        Activation function. Default is ``nn.PReLU()``.
    batchnorm : bool, optional
        If True a batch normalization layer is applied after every convolution. Default is ``False``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_convs: int = 3,
        activation: nn.Module = nn.PReLU(),
        batchnorm: bool = False,
    ):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of Conv2d.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Convoluted output.
        """
        if x.dim() == 5:
            x = x.squeeze(1)
            if x.shape[-1] == 2:
                x = x.permute(0, 3, 1, 2)
        return self.conv(x)  # type: ignore
