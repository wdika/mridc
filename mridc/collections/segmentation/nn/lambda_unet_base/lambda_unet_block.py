# coding=utf-8
__author__ = "Dimitrios Karkalousos"

import torch
from einops import rearrange
from torch import einsum, nn

from mridc.collections.reconstruction.nn.unet_base import unet_block


class LambdaLayer(nn.Module):
    """
    Implementation of a Lambda Layer of the Lambda UNet for MRI segmentation, as presented in [1].

    References
    ----------
    .. [1] Yanglan Ou, Ye Yuan, Xiaolei Huang, Kelvin Wong, John Volpi, James Z. Wang, Stephen T.C. Wong. LambdaUNet:
        2.5D Stroke Lesion Segmentation of Diffusion-weighted MR Images. 2021. https://arxiv.org/abs/2104.13917

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    query_depth : int, optional
        Number of channels for the keys. Default is ``16``.
    intra_depth : int, optional
        Number of neighboring slices. Default is ``1``.
    receptive_kernel : int, optional
        Local context kernel size. Default is ``3``.
    temporal_kernel : int, optional
        Temporal kernel. Default is ``1``.
    heads : int, optional
        Number of query heads. Default is ``4``.
    num_slices : int, optional
        Number of slices. Default is ``1``.
    """

    def __init__(  # noqa: R0913
        self,
        in_channels: int,
        out_channels: int,
        query_depth: int = 16,
        intra_depth: int = 1,
        receptive_kernel: int = 3,
        temporal_kernel: int = 1,
        heads: int = 4,
        num_slices: int = 1,
    ):
        super().__init__()
        self.dim_in = in_channels
        self.dim_out = out_channels

        self.q_depth = query_depth
        self.intra_depth = intra_depth

        if (out_channels % heads) != 0:
            raise AssertionError("out_channels must be divisible by number of heads for multi-head query.")
        self.v_depth = out_channels // heads
        self.heads = heads

        self.num_slices = num_slices

        self.receptive_kernel = receptive_kernel
        self.temporal_kernel = temporal_kernel

        self.to_q = nn.Sequential(
            nn.Conv2d(in_channels, query_depth * heads, kernel_size=1, bias=False),
            nn.BatchNorm2d(query_depth * heads),
        )
        self.to_k = nn.Sequential(
            nn.Conv2d(in_channels, query_depth * intra_depth, kernel_size=1, bias=False),
        )
        self.to_v = nn.Sequential(
            nn.Conv2d(in_channels, self.v_depth * intra_depth, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.v_depth * intra_depth),
        )

        if (receptive_kernel % 2) != 1:
            raise AssertionError("Receptive kernel size should be odd.")
        self.pos_conv = nn.Conv3d(
            intra_depth,
            query_depth,
            (1, receptive_kernel, receptive_kernel),
            padding=(0, receptive_kernel // 2, receptive_kernel // 2),
        )

        if temporal_kernel >= 3:
            if temporal_kernel > num_slices:
                raise AssertionError
            if (temporal_kernel % 2) != 1:
                raise AssertionError("Temporal kernel size should be odd.")
            self.temp_conv = nn.Conv2d(
                intra_depth,
                query_depth,
                (1, temporal_kernel),
                padding=(0, temporal_kernel // 2),
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: D102
        """Forward pass of the Lambda Layer."""
        batch, channel, height, width = (*input.shape,)  # type: ignore  # noqa: F841

        q = self.to_q(input)
        k = self.to_k(input)
        v = self.to_v(input)

        q = rearrange(q, "b (h k) hh ww -> b h k (hh ww)", h=self.heads)
        k = rearrange(k, "b (u k) hh ww -> b u k (hh ww)", u=self.intra_depth)
        v = rearrange(v, "b (u v) hh ww -> b u v (hh ww)", u=self.intra_depth)

        k = k.softmax(dim=-1)

        λc = einsum("b u k m, b u v m -> b k v", k, v)  # noqa: E741
        Yc = einsum("b h k n, b k v -> b h v n", q, λc)

        v_p = rearrange(v, "b u v (hh ww) -> b u v hh ww", hh=height, ww=width)  # type: ignore
        λp = self.pos_conv(v_p)  # noqa: E741
        Yp = einsum("b h k n, b k v n -> b h v n", q, λp.flatten(3))

        if self.temporal_kernel >= 3:
            v_t = rearrange(v, "(g t) u v p -> (g p) u v t", t=self.num_slices)
            λt = self.temp_conv(v_t)
            λt = rearrange(λt, "(g p) k v t -> (g t) k v p", p=height * width)  # type: ignore
            Yt = einsum("b h k n, b k v n -> b h v n", q, λt)
            Y = Yc + Yp + Yt
        else:
            Y = Yc + Yp

        return rearrange(Y, "b h v (hh ww) -> b (h v) hh ww", hh=height, ww=width)  # type: ignore


class LambdaBlock(nn.Module):
    """
    Implementation of a Lambda Black of the Lambda UNet for MRI segmentation, as presented in [1].

    References
    ----------
    .. [1] Yanglan Ou, Ye Yuan, Xiaolei Huang, Kelvin Wong, John Volpi, James Z. Wang, Stephen T.C. Wong. LambdaUNet:
        2.5D Stroke Lesion Segmentation of Diffusion-weighted MR Images. 2021. https://arxiv.org/abs/2104.13917

    Parameters
    ----------
    in_chans : int
        Number of input channels.
    out_chans : int
        Number of output channels.
    drop_prob : float
        Dropout probability.
    query_depth : int, optional
        Number of channels for the keys. Default is ``16``.
    intra_depth : int, optional
        Number of neighboring slices. Default is ``4``.
    receptive_kernel : int, optional
        Local context kernel size. Default is ``3``.
    temporal_kernel : int, optional
        Temporal kernel. Default is ``1``.
    num_slices : int, optional
        Number of slices. Default is ``1``.
    """

    def __init__(  # noqa: D107
        self,
        in_chans: int,
        out_chans: int,
        drop_prob: float,
        query_depth: int = 16,
        intra_depth: int = 4,
        receptive_kernel: int = 3,
        temporal_kernel: int = 1,
        num_slices: int = 1,
    ):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            LambdaLayer(
                in_chans,
                out_chans,
                query_depth=query_depth,
                intra_depth=intra_depth,
                receptive_kernel=receptive_kernel,
                temporal_kernel=temporal_kernel,
                heads=max(1, out_chans // 32),
                num_slices=num_slices,
            ),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            LambdaLayer(
                out_chans,
                out_chans,
                query_depth=query_depth,
                intra_depth=intra_depth,
                receptive_kernel=receptive_kernel,
                temporal_kernel=temporal_kernel,
                heads=max(1, out_chans // 32),
                num_slices=num_slices,
            ),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Lambda Block."""
        return self.layers(image)


class LambdaUNet(unet_block.Unet):
    """
    Implementation of an extended UNet with Lambda blocks, as presented in [1].

    References
    ----------
    .. [1] Yanglan Ou, Ye Yuan, Xiaolei Huang, Kelvin Wong, John Volpi, James Z. Wang, Stephen T.C. Wong. LambdaUNet:
        2.5D Stroke Lesion Segmentation of Diffusion-weighted MR Images. 2021. https://arxiv.org/abs/2104.13917

    Parameters
    ----------
    in_chans : int
        Number of input channels.
    out_chans : int
        Number of output channels.
    chans : int, optional
        Number of channels. Default is ``32``.
    num_pool_layers : int, optional
        Number of pooling layers. Default is ``4``.
    drop_prob : float
        Dropout probability. Default is ``0.0``.
    query_depth : int, optional
        Number of channels for the keys. Default is ``16``.
    intra_depth : int, optional
        Number of neighboring slices. Default is ``4``.
    receptive_kernel : int, optional
        Local context kernel size. Default is ``3``.
    temporal_kernel : int, optional
        Temporal kernel. Default is ``1``.
    num_slices : int, optional
        Number of slices. Default is ``1``.
    """

    def __init__(  # noqa: D107
        self,
        in_chans: int,
        out_chans: int,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
        query_depth: int = 16,
        intra_depth: int = 4,
        receptive_kernel: int = 3,
        temporal_kernel: int = 1,
        num_slices: int = 1,
    ):
        super().__init__(
            in_chans=in_chans, out_chans=out_chans, chans=chans, num_pool_layers=num_pool_layers, drop_prob=drop_prob
        )

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList(
            [
                LambdaBlock(
                    in_chans,
                    chans,
                    drop_prob=drop_prob,
                    query_depth=query_depth,
                    intra_depth=intra_depth,
                    receptive_kernel=receptive_kernel,
                    temporal_kernel=temporal_kernel,
                    num_slices=num_slices,
                )
            ]
        )
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(
                LambdaBlock(
                    ch,
                    ch * 2,
                    drop_prob=drop_prob,
                    query_depth=query_depth,
                    intra_depth=intra_depth,
                    receptive_kernel=receptive_kernel,
                    temporal_kernel=temporal_kernel,
                    num_slices=num_slices,
                )
            )
            ch *= 2
        self.conv = LambdaBlock(
            ch,
            ch * 2,
            drop_prob=drop_prob,
            query_depth=query_depth,
            intra_depth=intra_depth,
            receptive_kernel=receptive_kernel,
            temporal_kernel=temporal_kernel,
            num_slices=num_slices,
        )

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(unet_block.TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(
                LambdaBlock(
                    ch * 2,
                    ch,
                    drop_prob=drop_prob,
                    query_depth=query_depth,
                    intra_depth=intra_depth,
                    receptive_kernel=receptive_kernel,
                    temporal_kernel=temporal_kernel,
                    num_slices=num_slices,
                )
            )
            ch //= 2

        self.up_transpose_conv.append(unet_block.TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(
            nn.Sequential(
                LambdaBlock(
                    ch * 2,
                    ch,
                    drop_prob=drop_prob,
                    query_depth=query_depth,
                    intra_depth=intra_depth,
                    receptive_kernel=receptive_kernel,
                    temporal_kernel=temporal_kernel,
                    num_slices=num_slices,
                ),
                nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )
        )
