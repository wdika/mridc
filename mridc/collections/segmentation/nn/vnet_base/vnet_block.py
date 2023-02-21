# coding=utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/black0017/MedicalZooPytorch/blob/master/lib/medzoo/Vnet.py

import torch
from torch import nn


class LUConv(nn.Module):
    """
    LU Convolutional Block.

    Parameters
    ----------
    channels : int
        Number of channels.
    act : nn.Module
        Activation function.
    bias : bool
        Whether to use bias.

    .. note::
        This is a wrapper for Vnet implementation.
        See: https://github.com/black0017/MedicalZooPytorch/blob/master/lib/medzoo/Vnet.py
    """

    def __init__(self, channels: int, act: nn.Module = nn.ELU, bias: bool = False):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=5, padding=2, bias=bias),
            nn.BatchNorm2d(channels),
            act(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network."""
        return self.layers(x)


def _make_nconv(channels: int, depth: int, act: nn.Module = nn.ELU, bias: bool = False):
    """
    Make a stack of LUConv layers.

    Parameters
    ----------
    channels : int
        Number of channels.
    depth : int
        Number of LUConv layers.
    act : nn.Module
        Activation function.
    bias : bool
        Whether to use bias.

    Returns
    -------
    layers : nn.Sequential
        Stack of LUConv layers.

    .. note::
        This is a wrapper for Vnet implementation.
        See: https://github.com/black0017/MedicalZooPytorch/blob/master/lib/medzoo/Vnet.py
    """
    layers = [LUConv(channels=channels, act=act, bias=bias) for _ in range(depth)]
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    """
    Input Transition Block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    act : nn.Module
        Activation function.
    bias : bool
        Whether to use bias.

    .. note::
        This is a wrapper for Vnet implementation.
        See: https://github.com/black0017/MedicalZooPytorch/blob/master/lib/medzoo/Vnet.py
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 16,
        act: nn.Module = nn.ELU,
        bias: bool = False,
    ):
        super().__init__()

        if out_channels % in_channels != 0:
            raise ValueError(f"16 should be divisible by in_channels, got in_channels={in_channels}.")

        self.in_channels = in_channels
        self.act_function = act(inplace=True)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, bias=bias),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network."""
        out = self.conv_block(x)
        x16 = x.repeat(1, 16 // self.in_channels, 1, 1)
        out = self.act_function(out + x16)
        return out


class DownTransition(nn.Module):
    """
    Down Transition Block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    convs : int
        Number of LUConv layers.
    act : nn.Module
        Activation function.
    dropout_prob : float
        Dropout probability.
    bias : bool
        Whether to use bias.

    .. note::
        This is a wrapper for Vnet implementation.
        See: https://github.com/black0017/MedicalZooPytorch/blob/master/lib/medzoo/Vnet.py
    """

    def __init__(
        self,
        in_channels: int,
        convs: int,
        act: nn.Module = nn.ELU,
        dropout_prob: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()

        out_channels = 2 * in_channels
        self.down_conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act_function1 = act(inplace=True)
        self.act_function2 = act(inplace=True)
        self.ops = _make_nconv(out_channels, convs, act, bias)
        self.dropout = nn.Dropout2d(dropout_prob) if dropout_prob > 0.0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network."""
        down = self.act_function1(self.bn1(self.down_conv(x)))
        out = self.dropout(down) if self.dropout is not None else down
        out = self.ops(out)
        out = self.act_function2(out + down)
        return out


class UpTransition(nn.Module):
    """
    Up Transition Block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    convs : int
        Number of LUConv layers.
    act : nn.Module
        Activation function.
    dropout_prob : float
        Dropout probability.

    .. note::
        This is a wrapper for Vnet implementation.
        See: https://github.com/black0017/MedicalZooPytorch/blob/master/lib/medzoo/Vnet.py
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        convs: int,
        act: nn.Module = nn.ELU,
        dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels // 2, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(out_channels // 2)
        self.dropout = nn.Dropout2d(dropout_prob) if dropout_prob > 0.0 else None
        self.dropout2 = nn.Dropout2d(0.5)
        self.act_function1 = act(inplace=True)
        self.act_function2 = act(inplace=True)
        self.ops = _make_nconv(out_channels, convs, act)

    def forward(self, x: torch.Tensor, skipx: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network."""
        out = self.dropout(x) if self.dropout is not None else x
        skipxdo = self.dropout2(skipx)
        out = self.act_function1(self.bn1(self.up_conv(out)))
        xcat = torch.cat((out, skipxdo), 1)
        out = self.ops(xcat)
        out = self.act_function2(out + xcat)
        return out


class OutputTransition(nn.Module):
    """
    Output Transition Block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    act : nn.Module
        Activation function.
    bias : bool
        Whether to use bias.

    .. note::
        This is a wrapper for Vnet implementation.
        See: https://github.com/black0017/MedicalZooPytorch/blob/master/lib/medzoo/Vnet.py
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        act: nn.Module = nn.ELU,
        bias: bool = False,
    ):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, bias=bias),
            nn.BatchNorm2d(out_channels),
            act(inplace=True),
        )
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network."""
        # convolve 32 down to 2 channels
        out = self.conv_block(x)
        out = self.conv2(out)
        return out


class VNet(nn.Module):
    """
    Implementation of the V-Net for MRI segmentation, as presented in [1].

    References
    ----------
    .. [1] Fausto Milletari, Nassir Navab, Seyed-Ahmad Ahmadi. V-Net: Fully Convolutional Neural Networks for
        Volumetric Medical Image Segmentation, 2016. https://arxiv.org/abs/1606.04797

    Parameters
    ----------
    in_chans : int
        Number of input channels.
    out_chans : int
        Number of output channels.
    act : nn.Module
        Activation function.
    drop_prob : float
        Dropout probability.
    bias : bool
        Whether to use bias.

    .. note::
        This is a wrapper for Vnet implementation.
        See: https://github.com/black0017/MedicalZooPytorch/blob/master/lib/medzoo/Vnet.py
    """

    def __init__(
        self,
        in_chans: int = 1,
        out_chans: int = 1,
        act: str = "elu",
        drop_prob: float = 0.5,
        bias: bool = False,
    ):
        super().__init__()

        if act == "elu":
            act = nn.ELU
        elif act == "relu":
            act = nn.ReLU
        elif act == "prelu":
            act = nn.PReLU
        elif act == "leakyrelu":
            act = nn.LeakyReLU
        else:
            raise ValueError(
                f"Activation function {act} not supported. Please choose between ReLU, PReLU, LeakyReLU, ELU."
            )

        self.in_tr = InputTransition(in_chans, 16, act, bias=bias)
        self.down_tr32 = DownTransition(16, 1, act, bias=bias)
        self.down_tr64 = DownTransition(32, 2, act, bias=bias)
        self.down_tr128 = DownTransition(64, 3, act, dropout_prob=drop_prob, bias=bias)
        self.down_tr256 = DownTransition(128, 2, act, dropout_prob=drop_prob, bias=bias)
        self.up_tr256 = UpTransition(256, 256, 2, act, dropout_prob=drop_prob)
        self.up_tr128 = UpTransition(256, 128, 2, act, dropout_prob=drop_prob)
        self.up_tr64 = UpTransition(128, 64, 1, act)
        self.up_tr32 = UpTransition(64, 32, 1, act)
        self.out_tr = OutputTransition(32, out_chans, act, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network."""
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        x = self.up_tr256(out256, out128)
        x = self.up_tr128(x, out64)
        x = self.up_tr64(x, out32)
        x = self.up_tr32(x, out16)
        x = self.out_tr(x)

        return x
