# coding=utf-8
__author__ = "Dimitrios Karkalousos"

import math
from typing import List, Optional, Tuple

import torch
from torch import nn

import mridc.collections.reconstruction.models.unet_base.unet_block as unet_block


class DC(nn.Module):
    """Data consistency block."""

    def __init__(self, soft_dc: bool = False) -> None:
        super().__init__()
        self.soft_dc = soft_dc
        if self.soft_dc:
            self.dc_weight = nn.Parameter(torch.ones(1))

    def forward(self, kspace, og_kspace, mask=None):
        """Forward pass."""
        if mask is not None:
            zero = torch.zeros_like(kspace, device=kspace.device)
            dc = torch.where(mask.bool(), kspace - og_kspace, zero)
            if self.soft_dc:
                dc *= self.dc_weight
            return kspace - dc
        return kspace


class UnetEncoder(nn.Module):
    """UNet Encoder block, according to the implementation of the NormUnet"""

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        drop_prob: float = 0.0,
        padding_size: int = 15,
        normalize: bool = True,
        norm_groups: int = 2,
    ):
        """
        Parameters
        ----------
        chans : int
            Number of channels in the first layer.
        num_pools : int
            Number of down-sampling layers.
        in_chans : int, optional
            Number of input channels, by default 2
        drop_prob : float, optional
            Dropout probability, by default 0.0
        padding_size : int, optional
            Padding size, by default 15
        normalize : bool, optional
            Whether to normalize the input, by default True
        norm_groups : int, optional
            Number of groups for group normalization, by default 2
        """
        super().__init__()

        self.in_chans = in_chans
        self.chans = chans
        self.num_pools = num_pools
        self.drop_prob = drop_prob
        self.padding_size = padding_size
        self.normalize = normalize
        self.norm_groups = norm_groups

        self.down_sample_layers = torch.nn.ModuleList([unet_block.ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for _ in range(num_pools - 1):
            self.down_sample_layers.append(unet_block.ConvBlock(ch, ch * 2, drop_prob))
            ch *= 2
        self.conv = unet_block.ConvBlock(ch, ch * 2, drop_prob)

    @staticmethod
    def complex_to_chan_dim(x: torch.Tensor) -> torch.Tensor:
        """Convert the last dimension of the input to complex."""
        b, c, h, w, two = x.shape
        if two != 2:
            raise AssertionError
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize the input."""
        # group norm
        b, c, h, w = x.shape
        x = x.reshape(b, self.norm_groups, -1)
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        x = (x - mean) / std
        x = x.reshape(b, c, h, w)
        return x, mean, std

    def pad(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        """Pad the input with zeros to make it square."""
        _, _, h, w = x.shape
        w_mult = ((w - 1) | self.padding_size) + 1
        h_mult = ((h - 1) | self.padding_size) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        # TODO: fix this type when PyTorch fixes theirs
        # the documentation lies - this actually takes a list
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L3457
        # https://github.com/pytorch/pytorch/pull/16949
        x = torch.nn.functional.pad(x, w_pad + h_pad)
        return x, (h_pad, w_pad, h_mult, w_mult)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network."""
        iscomplex = False
        if x.shape[-1] == 2:
            x = self.complex_to_chan_dim(x)
            iscomplex = True

        mean = 1.0
        std = 1.0

        if self.normalize:
            x, mean, std = self.norm(x)

        x, pad_sizes = self.pad(x)

        stack = []
        output = x

        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = torch.nn.functional.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)
        stack.append(output)

        if self.normalize:
            return stack, iscomplex, pad_sizes, mean, std
        return stack, iscomplex, pad_sizes


class UnetDecoder(nn.Module):
    """UNet Decoder block, according to the implementation of the NormUnet"""

    def __init__(
        self,
        chans: int,
        num_pools: int,
        out_chans: int = 2,
        drop_prob: float = 0.0,
        padding_size: int = 15,
        normalize: bool = True,
        norm_groups: int = 2,
    ):
        """
        Parameters
        ----------
        chans : int
            Number of channels in the first layer.
        num_pools : int
            Number of down-sampling layers.
        out_chans : int, optional
            Number of output channels, by default 2
        drop_prob : float, optional
            Dropout probability, by default 0.0
        padding_size : int, optional
            Padding size, by default 15
        normalize : bool, optional
            Whether to normalize the input, by default True
        norm_groups : int, optional
            Number of groups for group normalization, by default 2
        """
        super().__init__()

        self.out_chans = out_chans
        self.chans = chans
        self.num_pools = num_pools
        self.drop_prob = drop_prob
        self.padding_size = padding_size
        self.normalize = normalize
        self.norm_groups = norm_groups

        ch = chans * (2 ** (num_pools - 1))
        self.up_conv = torch.nn.ModuleList()
        self.up_transpose_conv = torch.nn.ModuleList()
        for _ in range(num_pools - 1):
            self.up_transpose_conv.append(unet_block.TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(unet_block.ConvBlock(ch * 2, ch, drop_prob))
            ch //= 2

        self.up_transpose_conv.append(unet_block.TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(
            torch.nn.Sequential(
                unet_block.ConvBlock(ch * 2, ch, drop_prob),
                torch.nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )
        )

    @staticmethod
    def chan_complex_to_last_dim(x: torch.Tensor) -> torch.Tensor:
        """Convert the last dimension of the input to complex."""
        b, c2, h, w = x.shape
        if c2 % 2 != 0:
            raise AssertionError
        c = torch.div(c2, 2, rounding_mode="trunc")
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()

    @staticmethod
    def unpad(x: torch.Tensor, h_pad: List[int], w_pad: List[int], h_mult: int, w_mult: int) -> torch.Tensor:
        """Unpad the input."""
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]

    def unnorm(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Unnormalize the input."""
        b, c, h, w = x.shape
        input_data = x.reshape(b, self.norm_groups, -1)
        return (input_data * std + mean).reshape(b, c, h, w)

    def forward(self, x_stack, iscomplex=False, pad_sizes=None, mean=None, std=None):
        """Forward pass of the network."""
        output = x_stack.pop()
        # apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = x_stack.pop()
            output = transpose_conv(output)

            # reflect pad on the right/bottom if needed to handle odd input dimensions
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if torch.sum(torch.tensor(padding)) != 0:
                output = torch.nn.functional.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)

        if pad_sizes is not None:
            output = self.unpad(output, *pad_sizes)
        if self.normalize and mean is not None and std is not None:
            output = self.unnorm(output, mean, std)
        if iscomplex:
            output = self.chan_complex_to_last_dim(output)

        return output
