# coding=utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NKI-AI/direct/blob/main/direct/nn/didn/didn.py
# Copyright (c) DIRECT Contributors

import torch
import torch.nn as nn
import torch.nn.functional as F


class Subpixel(nn.Module):
    """
    Subpixel convolution layer for up-scaling of low resolution features at super-resolution as implemented in \
    Yu, Songhyun, et al.

    References
    ----------

    ..
        Yu, Songhyun, et al. “Deep Iterative Down-Up CNN for Image Denoising.” 2019 IEEE/CVF Conference on Computer \
        Vision and Pattern Recognition Workshops (CVPRW), 2019, pp. 2095–103. IEEE Xplore, \
        https://doi.org/10.1109/CVPRW.2019.00262.

    """

    def __init__(self, in_channels, out_channels, upscale_factor, kernel_size, padding=0):
        """
        Inits Subpixel.

        Parameters
        ----------
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        upscale_factor: Subpixel upscale factor.
        kernel_size: Convolution kernel size.
        padding: Padding size. Default: 0.
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels * upscale_factor**2, kernel_size=kernel_size, padding=padding
        )
        self.pixelshuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        """Computes Subpixel convolution on input torch.Tensor ``x``."""
        return self.pixelshuffle(self.conv(x))


class ReconBlock(nn.Module):
    """
    Reconstruction Block of DIDN model as implemented in Yu, Songhyun, et al.

    References
    ----------

    ..
        Yu, Songhyun, et al. “Deep Iterative Down-Up CNN for Image Denoising.” 2019 IEEE/CVF Conference on Computer \
        Vision and Pattern Recognition Workshops (CVPRW), 2019, pp. 2095–103. IEEE Xplore, \
        https://doi.org/10.1109/CVPRW.2019.00262.

    """

    def __init__(self, in_channels, num_convs):
        """
        Inits ReconBlock.

        Parameters
        ----------
        in_channels: Number of input channels.
        num_convs: Number of convolution blocks.
        """
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1),
                        nn.PReLU(),
                    ]
                )
                for _ in range(num_convs - 1)
            ]
        )
        self.convs.append(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1))
        self.num_convs = num_convs

    def forward(self, input_data):
        """
        Computes num_convs convolutions followed by PReLU activation on `input_data`.

        Parameters
        ----------
        input_data: Input tensor.
        """
        output = input_data.clone()
        for idx in range(self.num_convs):
            output = self.convs[idx](output)

        return input_data + output


class DUB(nn.Module):
    """
    Down-up block (DUB) for DIDN model as implemented in Yu, Songhyun, et al.

    References
    ----------

    ..
        Yu, Songhyun, et al. “Deep Iterative Down-Up CNN for Image Denoising.” 2019 IEEE/CVF Conference on Computer \
        Vision and Pattern Recognition Workshops (CVPRW), 2019, pp. 2095–103. IEEE Xplore, \
        https://doi.org/10.1109/CVPRW.2019.00262.

    """

    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        """
        Inits DUB.

        Parameters
        ----------
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Scale 1
        self.conv1_1 = nn.Sequential(*[nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1), nn.PReLU()] * 2)
        self.down1 = nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, stride=2, padding=1)
        # Scale 2
        self.conv2_1 = nn.Sequential(
            *[nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3, padding=1), nn.PReLU()]
        )
        self.down2 = nn.Conv2d(in_channels * 2, in_channels * 4, kernel_size=3, stride=2, padding=1)
        # Scale 3
        self.conv3_1 = nn.Sequential(
            *[
                nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size=3, padding=1),
                nn.PReLU(),
            ]
        )
        self.up1 = nn.Sequential(*[Subpixel(in_channels * 4, in_channels * 2, 2, 1, 0)])
        # Scale 2
        self.conv_agg_1 = nn.Conv2d(in_channels * 4, in_channels * 2, kernel_size=1)
        self.conv2_2 = nn.Sequential(
            *[
                nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3, padding=1),
                nn.PReLU(),
            ]
        )
        self.up2 = nn.Sequential(*[Subpixel(in_channels * 2, in_channels, 2, 1, 0)])
        # Scale 1
        self.conv_agg_2 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.conv1_2 = nn.Sequential(*[nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1), nn.PReLU()] * 2)
        self.conv_out = nn.Sequential(*[nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1), nn.PReLU()])

    @staticmethod
    def pad(x):
        """
        Pads input to height and width dimensions if odd.

        Parameters
        ----------
        x: Input to pad.

        Returns
        -------
        Padded tensor.
        """
        padding = [0, 0, 0, 0]

        if x.shape[-2] % 2 != 0:
            padding[3] = 1  # Padding right - width
        if x.shape[-1] % 2 != 0:
            padding[1] = 1  # Padding bottom - height
        if sum(padding) != 0:
            x = F.pad(x, padding, "reflect")
        return x

    @staticmethod
    def crop_to_shape(x, shape):
        """
        Crops ``x`` to specified shape.

        Parameters
        ----------
        x: Input tensor with shape (\*, H, W).
        shape: Crop shape corresponding to H, W.

        Returns
        -------
        Cropped tensor.
        """
        h, w = x.shape[-2:]

        if h > shape[0]:
            x = x[:, :, : shape[0], :]
        if w > shape[1]:
            x = x[:, :, :, : shape[1]]
        return x

    def forward(self, x):
        """
        Parameters
        ----------
        x: Input tensor.

        Returns
        -------
        DUB output.
        """
        x1 = self.pad(x.clone())
        x1 = x1 + self.conv1_1(x1)
        x2 = self.down1(x1)
        x2 = x2 + self.conv2_1(x2)
        out = self.down2(x2)
        out = out + self.conv3_1(out)
        out = self.up1(out)
        out = torch.cat([x2, self.crop_to_shape(out, x2.shape[-2:])], dim=1)
        out = self.conv_agg_1(out)
        out = out + self.conv2_2(out)
        out = self.up2(out)
        out = torch.cat([x1, self.crop_to_shape(out, x1.shape[-2:])], dim=1)
        out = self.conv_agg_2(out)
        out = out + self.conv1_2(out)
        out = x + self.crop_to_shape(self.conv_out(out), x.shape[-2:])
        return out


class DIDN(nn.Module):
    """
    Deep Iterative Down-up convolutional Neural network (DIDN) implementation as in Yu, Songhyun, et al.

    References
    ----------

    ..
        Yu, Songhyun, et al. “Deep Iterative Down-Up CNN for Image Denoising.” 2019 IEEE/CVF Conference on Computer \
        Vision and Pattern Recognition Workshops (CVPRW), 2019, pp. 2095–103. IEEE Xplore, \
        https://doi.org/10.1109/CVPRW.2019.00262.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 128,
        num_dubs: int = 6,
        num_convs_recon: int = 9,
        skip_connection: bool = False,
    ):
        """
        Inits DIDN.

        Parameters
        ----------
        in_channels: Number of input channels.
            int
        out_channels: Number of output channels.
            int
        hidden_channels: Number of hidden channels. First convolution out_channels.
            int, Default: 128.
        num_dubs: Number of DUB networks.
            int, Default: 6.
        num_convs_recon: Number of ReconBlock convolutions.
            int, Default: 9.
        skip_connection: Use skip connection.
            bool, Default: False.
        """
        super().__init__()
        self.conv_in = nn.Sequential(
            *[nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, padding=1), nn.PReLU()]
        )
        self.down = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.dubs = nn.ModuleList(
            [DUB(in_channels=hidden_channels, out_channels=hidden_channels) for _ in range(num_dubs)]
        )
        self.recon_block = ReconBlock(in_channels=hidden_channels, num_convs=num_convs_recon)
        self.recon_agg = nn.Conv2d(in_channels=hidden_channels * num_dubs, out_channels=hidden_channels, kernel_size=1)
        self.conv = nn.Sequential(
            *[
                nn.Conv2d(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.PReLU(),
            ]
        )
        self.up2 = Subpixel(hidden_channels, hidden_channels, 2, 1)
        self.conv_out = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )
        self.num_dubs = num_dubs
        self.skip_connection = (in_channels == out_channels) and skip_connection

    @staticmethod
    def crop_to_shape(x, shape):
        """
        Crops ``x`` to specified shape.

        Parameters
        ----------
        x: Input tensor with shape (\*, H, W).
        shape: Crop shape corresponding to H, W.

        Returns
        -------
        Cropped tensor.
        """
        h, w = x.shape[-2:]

        if h > shape[0]:
            x = x[:, :, : shape[0], :]
        if w > shape[1]:
            x = x[:, :, :, : shape[1]]
        return x

    def forward(self, x, channel_dim=1):
        """
        Takes as input a torch.Tensor `x` and computes DIDN(x).

        Parameters
        ----------
        x: Input tensor.
        channel_dim: Channel dimension. Default: 1.

        Returns
        -------
        DIDN output tensor.
        """
        out = self.conv_in(x)
        out = self.down(out)

        dub_outs = []
        for dub in self.dubs:
            out = dub(out)
            dub_outs.append(out)

        out = [self.recon_block(dub_out) for dub_out in dub_outs]
        out = self.recon_agg(torch.cat(out, dim=channel_dim))
        out = self.conv(out)
        out = self.up2(out)
        out = self.conv_out(out)
        out = self.crop_to_shape(out, x.shape[-2:])

        if self.skip_connection:
            out = x + out
        return out
