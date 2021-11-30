# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Parts of the code have been taken from https://github.com/facebookresearch/fastMRI
import math
from typing import List, Tuple

import torch


class NormUnet(torch.nn.Module):
    """
    Normalized U-Net model.

    This is the same as a regular U-Net, but with normalization applied to the input before the U-Net.
    This keeps the values more numerically stable during training.
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
        padding_size: int = 15,
        normalize: bool = True,
        norm_groups: int = 2,
    ):
        """

        Parameters
        ----------
        chans : Number of output channels of the first convolution layer.
        num_pools : Number of down-sampling and up-sampling layers.
        in_chans : Number of channels in the input to the U-Net model.
        out_chans : Number of channels in the output to the U-Net model.
        drop_prob : Dropout probability.
        padding_size: Size of the padding.
        normalize: Whether to normalize the input.
        norm_groups: Number of groups to use for group normalization.
        """
        super().__init__()

        self.unet = Unet(
            in_chans=in_chans, out_chans=out_chans, chans=chans, num_pool_layers=num_pools, drop_prob=drop_prob
        )

        self.padding_size = padding_size
        self.normalize = normalize

        self.norm_groups = norm_groups

    @staticmethod
    def complex_to_chan_dim(x: torch.Tensor) -> torch.Tensor:
        """Convert the last dimension of the input to complex."""
        b, c, h, w, two = x.shape
        if two != 2:
            raise AssertionError
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    @staticmethod
    def chan_complex_to_last_dim(x: torch.Tensor) -> torch.Tensor:
        """Convert the last dimension of the input to complex."""
        b, c2, h, w = x.shape
        if c2 % 2 != 0:
            raise AssertionError
        c = torch.div(c2, 2, rounding_mode="trunc")
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()

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

    def unnorm(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Unnormalize the input."""
        b, c, h, w = x.shape
        input_data = x.reshape(b, self.norm_groups, -1)
        return (input_data * std + mean).reshape(b, c, h, w)

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

    @staticmethod
    def unpad(x: torch.Tensor, h_pad: List[int], w_pad: List[int], h_mult: int, w_mult: int) -> torch.Tensor:
        """Unpad the input."""
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]

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
        x = self.unet(x)
        x = self.unpad(x, *pad_sizes)

        if self.normalize:
            x = self.unnorm(x, mean, std)

        if iscomplex:
            x = self.chan_complex_to_last_dim(x)

        return x


class Unet(torch.nn.Module):
    """
    PyTorch implementation of a U-Net model, as presented in [1]_.

    References
    ----------
    .. [1] O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention, pages 234–241. Springer, 2015.
    """

    def __init__(
        self, in_chans: int, out_chans: int, chans: int = 32, num_pool_layers: int = 4, drop_prob: float = 0.0
    ):
        """
        Parameters
        ----------
        in_chans: Number of channels in the input to the U-Net model.
        out_chans: Number of channels in the output to the U-Net model.
        chans: Number of output channels of the first convolution layer.
        num_pool_layers: Number of down-sampling and up-sampling layers.
        drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = torch.nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob))
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, drop_prob)

        self.up_conv = torch.nn.ModuleList()
        self.up_transpose_conv = torch.nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(
            torch.nn.Sequential(
                ConvBlock(ch * 2, ch, drop_prob), torch.nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1)
            )
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns
        -------
        Output tensor of shape `(N, out_chans, H, W)`.
        """
        stack = []
        output = image

        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = torch.nn.functional.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        # apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
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

        return output


class ConvBlock(torch.nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by instance normalization, LeakyReLU
    activation and dropout.
    """

    def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
        """
        Parameters
        ----------
        in_chans: Number of channels in the input.
        out_chans: Number of channels in the output.
        drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            torch.nn.InstanceNorm2d(out_chans),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
            torch.nn.Dropout2d(drop_prob),
            torch.nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            torch.nn.InstanceNorm2d(out_chans),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
            torch.nn.Dropout2d(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns
        -------
        Output tensor of shape `(N, out_chans, H, W)`.
        """
        return self.layers(image)


class TransposeConvBlock(torch.nn.Module):
    """
    A Transpose Convolutional Block that consists of one convolution transpose layers followed by instance
    normalization and LeakyReLU activation.
    """

    def __init__(self, in_chans: int, out_chans: int):
        """
        Parameters
        ----------
        in_chans: Number of channels in the input.
        out_chans: Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2, bias=False),
            torch.nn.InstanceNorm2d(out_chans),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns
        -------
        Output tensor of shape `(N, out_chans, H*2, W*2)`.
        """
        return self.layers(image)
