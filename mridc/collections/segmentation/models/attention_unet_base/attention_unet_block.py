# coding=utf-8
__author__ = "Dimitrios Karkalousos"

import torch
from torch import nn

import mridc.collections.reconstruction.models.unet_base.unet_block as unet_block


class AttentionGate(nn.Module):
    """A Convolutional Block that consists of two convolution layers each followed by instance normalization, \
    LeakyReLU, activation and dropout."""

    def __init__(self, in_chans_x: int, in_chans_g: int, out_chans: int):
        """
        Parameters
        ----------
        in_chans_x : int
            Number of input channels of the input tensor `x`.
        in_chans_g : int
            Number of input channels of the input tensor `g`.
        out_chans : int
            Number of output channels.
        """
        super().__init__()

        self.in_chans_x = in_chans_x
        self.in_chans_g = in_chans_g
        self.out_chans = out_chans

        self.W_x = nn.Sequential(nn.Conv2d(self.in_chans_x, out_chans, kernel_size=2, padding=0, stride=2, bias=False))
        self.W_g = nn.Sequential(nn.Conv2d(self.in_chans_g, out_chans, kernel_size=1, padding=0, bias=True))
        self.psi = nn.Sequential(nn.Conv2d(self.out_chans, 1, kernel_size=1, padding=0, bias=True))

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor `x` with shape [batch_size, in_chans_x, n_x, n_y].
        g : torch.Tensor
            Input tensor `g` with shape [batch_size, in_chans_g, n_x, n_y].

        Returns
        -------
        out : torch.Tensor
            Output tensor with shape [batch_size, out_chans, n_x, n_y].
        """
        W_x = self.W_x(x)
        w_g = self.W_g(g)
        W_g = nn.functional.interpolate(w_g, size=(W_x.shape[-2], W_x.shape[-1]), mode="bilinear", align_corners=False)
        f = nn.functional.relu(W_x + W_g, inplace=True)
        a = torch.sigmoid(self.psi(f))
        a = nn.functional.interpolate(a, size=(x.shape[-2], x.shape[-1]), mode="bilinear", align_corners=False)
        return a * x


class AttentionUnet(nn.Module):
    """
    Implementation of the Attention UNet, as presented in Ozan Oktay et al.

    References
    ----------
    ..

        O. Oktay, J. Schlemper, L.L. Folgoc, M. Lee, M. Heinrich, K. Misawa, K. Mori, S. McDonagh, N.Y. Hammerla, \
        B. Kainz, B. Glocker, D. Rueckert. Attention U-Net: Learning Where to Look for the Pancreas. \
        2018. https://arxiv.org/abs/1804.03999

    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
        block=unet_block.ConvBlock,
        **kwargs,
    ):
        """
        Parameters
        ----------
        in_chans : int
            Number of input channels.
        out_chans : int
            Number of output channels.
        chans : int
            Number of channels in the convolutional layers.
        num_pool_layers : int
            Number of pooling layers.
        drop_prob : float
            Dropout probability.
        block : nn.Module
            Convolutional block to use.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList([unet_block.ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(block(ch, ch * 2, drop_prob, **kwargs))
            ch *= 2
        self.conv = block(ch, ch * 2, drop_prob, **kwargs)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        self.up_attention_gates = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(unet_block.TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(unet_block.ConvBlock(ch * 2, ch, drop_prob))
            self.up_attention_gates.append(AttentionGate(ch, ch * 2, ch))
            ch //= 2

        self.up_transpose_conv.append(unet_block.TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(
            nn.Sequential(
                unet_block.ConvBlock(ch * 2, ch, drop_prob, **kwargs),
                nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )
        )
        self.up_attention_gates.append(AttentionGate(ch, ch * 2, ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape [batch_size, in_chans, n_x, n_y].

        Returns
        -------
        out : torch.Tensor
            Output tensor with shape [batch_size, out_chans, n_x, n_y].
        """
        stack = []
        output = x

        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = nn.functional.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        # apply up-sampling layers
        for transpose_conv, conv, attention_gate in zip(self.up_transpose_conv, self.up_conv, self.up_attention_gates):
            downsample_layer = stack.pop()
            downsample_layer = attention_gate(downsample_layer, output)
            output = transpose_conv(output)

            # reflect pad on the right/bottom if needed to handle odd input dimensions
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if torch.sum(torch.tensor(padding)) != 0:
                output = nn.functional.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)

        return output
