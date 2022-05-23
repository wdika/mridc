# coding=utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from:https://github.com/NKI-AI/direct/blob/main/direct/nn/multidomainnet/multidomain.py
# Copyright (c) DIRECT Contributors

import torch
import torch.nn as nn
import torch.nn.functional as F

from mridc.collections.common.parts.fft import fft2c, ifft2c
from mridc.collections.common.parts.utils import complex_conj, complex_mul


class MultiDomainConv2d(nn.Module):
    """Multi-domain convolution layer."""

    def __init__(
        self,
        fft_type,
        in_channels,
        out_channels,
        **kwargs,
    ):
        super().__init__()

        self.image_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels // 2, **kwargs)
        self.kspace_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels // 2, **kwargs)
        self.fft_type = fft_type
        self._channels_dim = 1
        self._spatial_dims = [1, 2]

    def forward(self, image):
        """Forward method for the MultiDomainConv2d class."""
        kspace = [
            fft2c(im, fft_type=self.fft_type, fft_dim=self._spatial_dims)
            for im in torch.split(image.permute(0, 2, 3, 1).contiguous(), 2, -1)
        ]
        kspace = torch.cat(kspace, -1).permute(0, 3, 1, 2)
        kspace = self.kspace_conv(kspace)

        backward = [
            ifft2c(ks.float(), fft_type=self.fft_type, fft_dim=self._spatial_dims).type(image.type())
            for ks in torch.split(kspace.permute(0, 2, 3, 1).contiguous(), 2, -1)
        ]
        backward = torch.cat(backward, -1).permute(0, 3, 1, 2)

        image = self.image_conv(image)
        image = torch.cat([image, backward], dim=self._channels_dim)
        return image


class MultiDomainConvTranspose2d(nn.Module):
    """Multi-Domain convolutional transpose layer."""

    def __init__(
        self,
        fft_type,
        in_channels,
        out_channels,
        **kwargs,
    ):
        super().__init__()

        self.image_conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels // 2, **kwargs)
        self.kspace_conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels // 2, **kwargs)
        self.fft_type = fft_type
        self._channels_dim = 1
        self._spatial_dims = [1, 2]

    def forward(self, image):
        """Forward method for the MultiDomainConvTranspose2d class."""
        kspace = [
            fft2c(im, fft_type=self.fft_type, fft_dim=self._spatial_dims)
            for im in torch.split(image.permute(0, 2, 3, 1).contiguous(), 2, -1)
        ]
        kspace = torch.cat(kspace, -1).permute(0, 3, 1, 2)
        kspace = self.kspace_conv(kspace)

        backward = [
            ifft2c(ks.float(), fft_type=self.fft_type, fft_dim=self._spatial_dims).type(image.type())
            for ks in torch.split(kspace.permute(0, 2, 3, 1).contiguous(), 2, -1)
        ]
        backward = torch.cat(backward, -1).permute(0, 3, 1, 2)

        image = self.image_conv(image)
        return torch.cat([image, backward], dim=self._channels_dim)


class MultiDomainConvBlock(nn.Module):
    """
    A multi-domain convolutional block that consists of two multi-domain convolution layers each followed by instance
    normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, fft_type, in_channels: int, out_channels: int, dropout_probability: float):
        """
        Parameters
        ----------
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        dropout_probability: Dropout probability.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_probability = dropout_probability

        self.layers = nn.Sequential(
            MultiDomainConv2d(fft_type, in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(dropout_probability),
            MultiDomainConv2d(fft_type, out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(dropout_probability),
        )

    def forward(self, _input: torch.Tensor):
        """Forward method for the MultiDomainConvBlock class."""
        return self.layers(_input)

    def __repr__(self):
        return (
            f"MultiDomainConvBlock(in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"dropout_probability={self.dropout_probability})"
        )


class TransposeMultiDomainConvBlock(nn.Module):
    """
    A Transpose Convolutional Block that consists of one convolution transpose layers followed by instance
    normalization and LeakyReLU activation.
    """

    def __init__(self, fft_type, in_channels: int, out_channels: int):
        """
        Parameters
        ----------
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers = nn.Sequential(
            MultiDomainConvTranspose2d(fft_type, in_channels, out_channels, kernel_size=2, stride=2, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, input_data: torch.Tensor):
        """Forward method for the TransposeMultiDomainConvBlock class."""
        return self.layers(input_data)

    def __repr__(self):
        return f"MultiDomainConvBlock(in_channels={self.in_channels}, out_channels={self.out_channels})"


class StandardizationLayer(nn.Module):
    """
    Multi-channel data standardization method. Inspired by AIRS model submission to the Fast MRI 2020 challenge.
    Given individual coil images :math:`\{x_i\}_{i=1}^{N_c}` and sensitivity coil maps :math:`\{S_i\}_{i=1}^{N_c}`

    it returns
    .. math::
        [(x_{\text{sense}}, {x_{\text{res}}}_1), ..., (x_{\text{sense}}, {x_{\text{res}}}_{N_c})]
    where :math:`{x_{\text{res}}}_i = xi - S_i \times x_{\text{sense}}` and
    :math:`x_{\text{sense}} = \sum_{i=1}^{N_c} {S_i}^{*} \times x_i`.
    """

    def __init__(self, coil_dim=1, channel_dim=-1):
        super().__init__()
        self.coil_dim = coil_dim
        self.channel_dim = channel_dim

    def forward(self, coil_images: torch.Tensor, sensitivity_map: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        combined_image = complex_mul(coil_images, complex_conj(sensitivity_map)).sum(self.coil_dim)
        residual_image = combined_image.unsqueeze(self.coil_dim) - complex_mul(
            combined_image.unsqueeze(self.coil_dim), sensitivity_map
        )
        return torch.cat(
            [
                torch.cat(
                    [combined_image, residual_image.select(self.coil_dim, idx)],
                    self.channel_dim,
                ).unsqueeze(self.coil_dim)
                for idx in range(coil_images.size(self.coil_dim))
            ],
            self.coil_dim,
        )


class MultiDomainUnet2d(nn.Module):
    """
    Unet modification to be used with Multi-domain network as in AIRS Medical submission to the Fast MRI 2020
    challenge.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_filters: int,
        num_pool_layers: int,
        dropout_probability: float,
        fft_type: str = "orthogonal",
    ):
        """
        Parameters
        ----------
        in_channels: Number of input channels to the u-net.
        out_channels: Number of output channels to the u-net.
        num_filters: Number of output channels of the first convolutional layer.
        num_pool_layers: Number of down-sampling and up-sampling layers (depth).
        dropout_probability: Dropout probability.
        fft_type: FFT type.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_filters = num_filters
        self.num_pool_layers = num_pool_layers
        self.dropout_probability = dropout_probability
        self.fft_type = fft_type

        self.down_sample_layers = nn.ModuleList(
            [MultiDomainConvBlock(fft_type, in_channels, num_filters, dropout_probability)]
        )
        ch = num_filters
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers += [MultiDomainConvBlock(fft_type, ch, ch * 2, dropout_probability)]
            ch *= 2
        self.conv = MultiDomainConvBlock(fft_type, ch, ch * 2, dropout_probability)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv += [TransposeMultiDomainConvBlock(fft_type, ch * 2, ch)]
            self.up_conv += [MultiDomainConvBlock(fft_type, ch * 2, ch, dropout_probability)]
            ch //= 2

        self.up_transpose_conv += [TransposeMultiDomainConvBlock(fft_type, ch * 2, ch)]
        self.up_conv += [
            nn.Sequential(
                MultiDomainConvBlock(fft_type, ch * 2, ch, dropout_probability),
                nn.Conv2d(ch, self.out_channels, kernel_size=1, stride=1),
            )
        ]

    def forward(self, input_data: torch.Tensor):
        """Forward pass of the u-net."""
        stack = []
        output = input_data

        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        # Apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)

            # Reflect pad on the right/bottom if needed to handle odd input dimensions.
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # Padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # Padding bottom
            if sum(padding) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)

        return output
