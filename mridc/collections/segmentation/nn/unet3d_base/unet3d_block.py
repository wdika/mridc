# coding=utf-8
__author__ = "Dimitrios Karkalousos"

import torch
from torch import nn


class Conv3dBlock(nn.Module):
    """A 3D convolutional block."""

    def __init__(self, in_chans: int, out_chans: int, drop_prob: float, **kwargs):  # noqa: D107
        """
        Parameters
        ----------
        in_chans : int
            Number of input channels.
        out_chans : int
            Number of output channels.
        drop_prob : float
            Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv3d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout3d(drop_prob),
            nn.Conv3d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout3d(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network."""
        return self.layers(image)


class TransposeConv3dBlock(nn.Module):
    """A 3D transposed convolutional block."""

    def __init__(self, in_chans: int, out_chans: int):
        """
        Parameters
        ----------
        in_chans : int
            Number of input channels.
        out_chans : int
            Number of output channels.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.ConvTranspose3d(in_chans, out_chans, kernel_size=(1, 2, 2), stride=(1, 2, 2), bias=False),
            nn.InstanceNorm3d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network."""
        return self.layers(image)


class UNet3D(nn.Module):
    """
    Implementation of the (3D) UNet for MRI segmentation, as presented in [1].

    References
    ----------
    .. [1] O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image
        segmentation. In International Conference on Medical image computing and computer-assisted intervention, pages
        234–241. Springer, 2015.

    Parameters
    ----------
    in_chans : int
        Number of input channels.
    out_chans : int
        Number of output channels.
    chans : int
        Number of output channels of the first convolutional layer. Default is ``32``.
    num_pool_layers : int
        Number of down-sampling and up-sampling layers. Default is ``4``.
    drop_prob : float
        Dropout probability. Default is ``0.0``.
    block : nn.Module
        Convolutional block to use. Default is ``Conv3dBlock``.
    """

    def __init__(  # noqa: D107
        self,
        in_chans: int,
        out_chans: int,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
        block=Conv3dBlock,
        **kwargs,
    ):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList([Conv3dBlock(in_chans, chans, drop_prob)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(block(ch, ch * 2, drop_prob, **kwargs))
            ch *= 2
        self.conv = block(ch, ch * 2, drop_prob, **kwargs)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConv3dBlock(ch * 2, ch))
            self.up_conv.append(Conv3dBlock(ch * 2, ch, drop_prob))
            ch //= 2

        self.up_transpose_conv.append(TransposeConv3dBlock(ch * 2, ch))
        self.up_conv.append(
            nn.Sequential(
                Conv3dBlock(ch * 2, ch, drop_prob, **kwargs),
                nn.Conv3d(ch, self.out_chans, kernel_size=1, stride=1),
            )
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network."""
        stack = []
        output = image

        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = nn.functional.avg_pool3d(output, kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0)

        output = self.conv(output)

        # apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)

            # reflect pad on the right/bottom if needed to handle odd input dimensions
            padding = [0, 0, 0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if output.shape[-3] != downsample_layer.shape[-3]:
                padding[5] = 1  # padding back
            if torch.sum(torch.tensor(padding)) != 0:
                output = nn.functional.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)

        return output
