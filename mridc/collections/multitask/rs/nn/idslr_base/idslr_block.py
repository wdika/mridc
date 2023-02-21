# coding=utf-8
__author__ = "Dimitrios Karkalousos"

import math
from typing import List, Tuple

import torch
from torch import Tensor, nn

import mridc.collections.reconstruction.nn.unet_base.unet_block as unet_block


class DC(nn.Module):
    """
    IDSLR Data consistency block, as presented in [1].

    References
    ----------
    .. [1] Pramanik A, Wu X, Jacob M. Joint calibrationless reconstruction and segmentation of parallel MRI. arXiv
        preprint arXiv:2105.09220. 2021 May 19.
    """

    def __init__(self):
        super().__init__()
        self.dc_weight = nn.Parameter(torch.ones(1))

    def forward(self, prediction_kspace: Tensor, reference_kspace: Tensor, mask: Tensor) -> Tensor:
        """
        Forward pass of the DC block.

        Parameters
        ----------
        prediction_kspace : torch.Tensor
            Prediction k-space. Shape: (batch, channels, height, width, complex)
        reference_kspace : torch.Tensor
            Reference k-space. Shape: (batch, channels, height, width, complex)
        mask : torch.Tensor
            Subsampling mask. Shape: (batch, channels, height, width, 1)

        Returns
        -------
        torch.Tensor
            Data consistency k-space. Shape: (batch, channels, height, width, complex)
        """
        return torch.div(
            torch.view_as_complex(reference_kspace) + self.dc_weight * torch.view_as_complex(prediction_kspace),
            mask.squeeze(-1) + torch.complex(self.dc_weight, torch.zeros_like(self.dc_weight)),
        )


class UnetEncoder(nn.Module):
    """
    UNet Encoder block, according to the implementation of the NormUnet.

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
    normalize : bool, optional
        Whether to normalize the input, by default True
    padding : bool, optional
        Whether to pad the input, by default True
    padding_size : int, optional
        Padding size, by default 15
    norm_groups : int, optional
        Number of groups for group normalization, by default 2
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        drop_prob: float = 0.0,
        normalize: bool = True,
        padding: bool = True,
        padding_size: int = 15,
        norm_groups: int = 2,
    ):
        super().__init__()

        self.in_chans = in_chans
        self.chans = chans
        self.num_pools = num_pools
        self.drop_prob = drop_prob
        self.normalize = normalize
        self.padding = padding
        self.padding_size = padding_size
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

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], bool, Tuple[List[int], List[int], int, int], torch.Tensor, torch.Tensor]:
        """
        Forward pass of the network.

        Parameters
        ----------
        x : torch.Tensor
            Input data. Shape: (batch, channels, height, width, complex)

        Returns
        -------
        List[torch.Tensor]
            List of down-sampled layers.
        bool
            Whether the input was complex.
        Tuple[List[int], List[int], int, int]
            Padding sizes.
        torch.Tensor
            Mean of the input.
        torch.Tensor
            Standard deviation of the input.
        """
        iscomplex = False
        if x.shape[-1] == 2:
            x = self.complex_to_chan_dim(x)
            iscomplex = True

        if self.normalize:
            x, mean, std = self.norm(x)
        else:
            mean = 1.0
            std = 1.0

        if self.padding:
            x, pad_sizes = self.pad(x)
        else:
            pad_sizes = None

        stack = []
        output = x

        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = torch.nn.functional.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)
        stack.append(output)

        return stack, iscomplex, pad_sizes, mean, std  # type: ignore


class UnetDecoder(nn.Module):
    """
    UNet Encoder block, according to the implementation of the NormUnet.

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
    normalize : bool, optional
        Whether to normalize the input, by default True
    padding : bool, optional
        Whether to pad the input, by default True
    padding_size : int, optional
        Padding size, by default 15
    norm_groups : int, optional
        Number of groups for group normalization, by default 2
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        out_chans: int = 2,
        drop_prob: float = 0.0,
        normalize: bool = True,
        padding: bool = True,
        padding_size: int = 15,
        norm_groups: int = 2,
    ):
        super().__init__()

        self.out_chans = out_chans
        self.chans = chans
        self.num_pools = num_pools
        self.drop_prob = drop_prob
        self.normalize = normalize
        self.padding = padding
        self.padding_size = padding_size
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

    def forward(
        self,
        x_stack: List[torch.Tensor],
        iscomplex: bool = False,
        pad_sizes: Tuple[List[int], List[int], int, int] = None,
        mean: torch.Tensor = None,
        std: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass of the network.

        Parameters
        ----------
        x_stack : List[torch.Tensor]
            List of tensors from the encoder.
        iscomplex : bool, optional
            Whether the input is complex. Default is ``False``.
        pad_sizes : Tuple[List[int], List[int], int, int], optional
            Padding sizes. Default is ``None``.
        mean : torch.Tensor, optional
            Mean of the input. Default is ``None``.
        std : torch.Tensor, optional
            Standard deviation of the input. Default is ``None``.

        Returns
        -------
        torch.Tensor
            Output of the network.
        """
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

        if self.padding:
            output = self.unpad(output, *pad_sizes)  # type: ignore
        if self.normalize and mean is not None and std is not None:
            output = self.unnorm(output, mean, std)
        if iscomplex:
            output = self.chan_complex_to_last_dim(output)

        return output
