# coding=utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/unetr.py

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn

from mridc.collections.segmentation.nn.vit_base.utils import get_conv_layer
from mridc.collections.segmentation.nn.vit_base.vit_block import ViT


class UnetOutBlock(nn.Module):
    """
    Implementation of the output block of UNETR, as presented in [1].

    References
    ----------
    .. Hatamizadeh A, Tang Y, Nath V, Yang D, Myronenko A, Landman B, Roth HR, Xu D. Unetr: Transformers for 3d medical
        image segmentation. InProceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision 2022
        (pp. 574-584).

    Parameters
    ----------
    spatial_dims : int
        Number of spatial dimensions of the input image.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    dropout : Optional[Union[Tuple, str, float]]
        Dropout rate.

    .. note::
        This is a wrapper for monai implementation of UNETR.
        See: https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/unetr.py
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        dropout: Optional[Union[Tuple, str, float]] = None,
    ):
        super().__init__()
        self.conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            dropout=dropout,
            bias=True,
            act=None,
            norm=None,
            conv_only=False,
        )

    def forward(self, inp):
        return self.conv(inp)


class UnetrBasicBlock(nn.Module):
    """
    A CNN module that can be used for UNETR, as presented in [1].

    References
    ----------
    .. Hatamizadeh A, Tang Y, Nath V, Yang D, Myronenko A, Landman B, Roth HR, Xu D. Unetr: Transformers for 3d medical
        image segmentation. InProceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision 2022
        (pp. 574-584).

    Parameters
    ----------
    spatial_dims : int
        Number of spatial dimensions of the input image.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : Union[Sequence[int], int]
        Convolution kernel size.
    stride : Union[Sequence[int], int]
        Convolution stride.
    norm_name : Union[Tuple, str]
        Feature normalization type and arguments.
    res_block : bool
        If True, use a residual block.

    .. note::
        This is a wrapper for monai implementation of UNETR.
        See: https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/unetr.py
    """

    def __init__(  # noqa: C901
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        res_block: bool = False,
    ):
        super().__init__()

        if res_block:
            self.layer = UnetResBlock(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                norm_name=norm_name,
            )
        else:
            self.layer = UnetBasicBlock(  # type: ignore
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                norm_name=norm_name,
            )

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Forward function of the module."""
        return self.layer(inp)


class UnetrPrUpBlock(nn.Module):
    """
    A projection upsampling module that can be used for UNETR, as presented in [1].

    References
    ----------
    .. Hatamizadeh A, Tang Y, Nath V, Yang D, Myronenko A, Landman B, Roth HR, Xu D. Unetr: Transformers for 3d medical
        image segmentation. InProceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision 2022
        (pp. 574-584).

    Parameters
    ----------
    spatial_dims : int
        Number of spatial dimensions of the input image.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    num_layer : int
        Number of layers.
    kernel_size : Union[Sequence[int], int]
        Convolution kernel size.
    stride : Union[Sequence[int], int]
        Convolution stride.
    upsample_kernel_size : Union[Sequence[int], int]
        Upsampling kernel size.
    norm_name : Union[Tuple, str]
        Feature normalization type and arguments.
    conv_block : bool
        If True, use a convolution block.
    res_block : bool
        If True, use a residual block.

    .. note::
        This is a wrapper for monai implementation of UNETR.
        See: https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/unetr.py
    """

    def __init__(  # noqa: C901
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_layer: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        conv_block: bool = False,
        res_block: bool = False,
    ):
        super().__init__()

        upsample_stride = upsample_kernel_size
        self.transp_conv_init = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )
        if conv_block:
            if res_block:
                self.blocks = nn.ModuleList(
                    [
                        nn.Sequential(
                            get_conv_layer(
                                spatial_dims,
                                out_channels,
                                out_channels,
                                kernel_size=upsample_kernel_size,
                                stride=upsample_stride,
                                conv_only=True,
                                is_transposed=True,
                            ),
                            UnetResBlock(
                                spatial_dims=spatial_dims,
                                in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                norm_name=norm_name,
                            ),
                        )
                        for i in range(num_layer)
                    ]
                )
            else:
                self.blocks = nn.ModuleList(
                    [
                        nn.Sequential(
                            get_conv_layer(
                                spatial_dims,
                                out_channels,
                                out_channels,
                                kernel_size=upsample_kernel_size,
                                stride=upsample_stride,
                                conv_only=True,
                                is_transposed=True,
                            ),
                            UnetBasicBlock(
                                spatial_dims=spatial_dims,
                                in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                norm_name=norm_name,
                            ),
                        )
                        for i in range(num_layer)
                    ]
                )
        else:
            self.blocks = nn.ModuleList(
                [
                    get_conv_layer(
                        spatial_dims,
                        out_channels,
                        out_channels,
                        kernel_size=upsample_kernel_size,
                        stride=upsample_stride,
                        conv_only=True,
                        is_transposed=True,
                    )
                    for i in range(num_layer)
                ]
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function of the module."""
        x = self.transp_conv_init(x)
        for blk in self.blocks:
            x = blk(x)
        return x


class UnetrUpBlock(nn.Module):
    """
    An upsampling module that can be used for UNETR, as presented in [1].

    References
    ----------
    .. Hatamizadeh A, Tang Y, Nath V, Yang D, Myronenko A, Landman B, Roth HR, Xu D. Unetr: Transformers for 3d medical
        image segmentation. InProceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision 2022
        (pp. 574-584).

    Parameters
    ----------
    spatial_dims : int
        Number of spatial dimensions of the input image.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : Union[Sequence[int], int]
        Convolution kernel size.
    upsample_kernel_size : Union[Sequence[int], int]
        Upsampling kernel size.
    norm_name : Union[Tuple, str]
        Feature normalization type and arguments.
    res_block : bool
        If True, use a residual block.

    .. note::
        This is a wrapper for monai implementation of UNETR.
        See: https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/unetr.py
    """

    def __init__(  # noqa: C901
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        res_block: bool = False,
    ):
        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        if res_block:
            self.conv_block = UnetResBlock(
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )
        else:
            self.conv_block = UnetBasicBlock(  # type: ignore
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )

    def forward(self, inp: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Forward function of the module."""
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)
        out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)
        return out


class UnetResBlock(nn.Module):
    """
    A skip-connection based module for UNETR, as presented in [1].

    References
    ----------
    .. Hatamizadeh A, Tang Y, Nath V, Yang D, Myronenko A, Landman B, Roth HR, Xu D. Unetr: Transformers for 3d medical
        image segmentation. InProceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision 2022
        (pp. 574-584).

    Parameters
    ----------
    spatial_dims : int
        Number of spatial dimensions of the input image.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : Union[Sequence[int], int]
        Convolution kernel size.
    stride : Union[Sequence[int], int]
        Convolution stride.
    norm_name : Union[Tuple, str]
        Feature normalization type and arguments.
    act_name : Union[Tuple, str]
        Activation function type and arguments.
    dropout : Optional[Union[Tuple, str, float]]
        Dropout rate.
    """

    def __init__(  # noqa: C901
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],  # noqa: A002
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),  # noqa: A002
        dropout: Optional[Union[Tuple, str, float]] = None,
    ):
        super().__init__()
        self.conv1 = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dropout=dropout,
            act=None,
            norm=None,
            conv_only=False,
        )
        self.conv2 = get_conv_layer(
            spatial_dims,
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout=dropout,
            act=None,
            norm=None,
            conv_only=False,
        )
        self.lrelu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        if spatial_dims == 2:
            self.norm1 = nn.InstanceNorm2d(out_channels)
            self.norm2 = nn.InstanceNorm2d(out_channels)
            self.norm3 = nn.InstanceNorm2d(out_channels)
        elif spatial_dims == 3:
            self.norm1 = nn.InstanceNorm3d(out_channels)
            self.norm2 = nn.InstanceNorm3d(out_channels)
            self.norm3 = nn.InstanceNorm3d(out_channels)
        self.downsample = in_channels != out_channels
        stride_np = np.atleast_1d(stride)
        if not np.all(stride_np == 1):
            self.downsample = True
        if self.downsample:
            self.conv3 = get_conv_layer(
                spatial_dims,
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                dropout=dropout,
                act=None,
                norm=None,
                conv_only=False,
            )

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Forward function of the module."""
        residual = inp
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if hasattr(self, "conv3"):
            residual = self.conv3(residual)
        if hasattr(self, "norm3"):
            residual = self.norm3(residual)
        out += residual
        out = self.lrelu(out)
        return out


class UnetUpBlock(nn.Module):
    """
    An upsampling module that can be used for UNETR, as presented in [1].

    References
    ----------
    .. Hatamizadeh A, Tang Y, Nath V, Yang D, Myronenko A, Landman B, Roth HR, Xu D. Unetr: Transformers for 3d medical
        image segmentation. InProceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision 2022
        (pp. 574-584).

    Parameters
    ----------
    spatial_dims : int
        Number of spatial dimensions of the input image.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : Union[Sequence[int], int]
        Convolution kernel size.
    stride : Union[Sequence[int], int]
        Convolution stride.
    upsample_kernel_size : Union[Sequence[int], int]
        Upsampling kernel size.
    norm_name : Union[Tuple, str]
        Feature normalization type and arguments.
    act_name : Union[Tuple, str]
        Activation function type and arguments.
    dropout : Optional[Union[Tuple, str, float]]
        Dropout rate.
    trans_bias : bool
        Whether to use bias in the transposed convolution layer. Default is ``False``.

    .. note::
        This is a wrapper for monai implementation of UNETR.
        See: https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/unetr.py
    """

    def __init__(  # noqa: C901
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],  # noqa: A002
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[Tuple, str, float]] = None,
        trans_bias: bool = False,
    ):
        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            dropout=dropout,
            bias=trans_bias,
            act=None,
            norm=None,
            conv_only=False,
            is_transposed=True,
        )
        self.conv_block = UnetBasicBlock(
            spatial_dims,
            out_channels + out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout=dropout,
            norm_name=norm_name,
            act_name=act_name,
        )

    def forward(self, inp: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Forward function of the module."""
        # number of channels for skip should equal to out_channels
        out = self.transp_conv(inp)
        out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)
        return out


class UnetBasicBlock(nn.Module):
    """
    A CNN module that can be used for UNETR, as presented in [1].

    References
    ----------
    .. Hatamizadeh A, Tang Y, Nath V, Yang D, Myronenko A, Landman B, Roth HR, Xu D. Unetr: Transformers for 3d medical
        image segmentation. InProceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision 2022
        (pp. 574-584).

    Parameters
    ----------
    spatial_dims : int
        Number of spatial dimensions of the input image.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : Union[Sequence[int], int]
        Convolution kernel size.
    stride : Union[Sequence[int], int]
        Convolution stride.
    norm_name : Union[Tuple, str]
        Feature normalization type and arguments.
    act_name : Union[Tuple, str]
        Activation function type and arguments.
    dropout : Optional[Union[Tuple, str, float]]
        Dropout rate.

    .. note::
        This is a wrapper for monai implementation of UNETR.
        See: https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/unetr.py
    """

    def __init__(  # noqa: C901
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],  # noqa: A002
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),  # noqa: A002
        dropout: Optional[Union[Tuple, str, float]] = None,
    ):
        super().__init__()
        self.conv1 = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dropout=dropout,
            act=None,
            norm=None,
            conv_only=False,
        )
        self.conv2 = get_conv_layer(
            spatial_dims,
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout=dropout,
            act=None,
            norm=None,
            conv_only=False,
        )
        self.lrelu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        if spatial_dims == 2:
            self.norm1 = nn.InstanceNorm2d(out_channels)
            self.norm2 = nn.InstanceNorm2d(out_channels)
        elif spatial_dims == 3:
            self.norm1 = nn.InstanceNorm3d(out_channels)
            self.norm2 = nn.InstanceNorm3d(out_channels)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Forward function of the module."""
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.lrelu(out)
        return out


class UNETR(nn.Module):
    """
    UNETR as presented in [1].

    References
    ----------
    .. Hatamizadeh A, Tang Y, Nath V, Yang D, Myronenko A, Landman B, Roth HR, Xu D. Unetr: Transformers for 3d medical
        image segmentation. InProceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision 2022
        (pp. 574-584).

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    img_size : Union[Sequence[int], int]
        Dimension of input image.
    feature_size : int
        Dimension of network feature size. Default is ``16``.
    hidden_size : int
        Dimension of network hidden size. Default is ``768``.
    mlp_dim : int
        Dimension of network mlp size. Default is ``3072``.
    num_heads : int
        Number of attention heads. Default is ``12``.
    pos_embed : str
        Positional embedding type. Default is ``"conv"``.
    norm_name : Union[Tuple, str]
        Feature normalization type and arguments. Default is ``"instance"``.
    conv_block : bool
        Whether to use convolutional block. Default is ``True``.
    res_block : bool
        Whether to use residual block. Default is ``True``.
    dropout_rate : float
        Dropout rate. Default is ``0.0``.
    spatial_dims : int
        Number of spatial dimensions of the input image. Default is ``3``.
    qkv_bias : bool
        Whether to use bias for qkv. Default is ``False``.

    .. note::
        This is a wrapper for monai implementation of UNETR.
        See: https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/unetr.py
    """

    def __init__(  # noqa: C901
        self,
        in_channels: int,
        out_channels: int,
        img_size: Union[Sequence[int], int],
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "conv",
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        qkv_bias: bool = False,
    ):
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.num_layers = 12
        # img_size = (img_size,) * spatial_dims  # type: ignore
        self.patch_size = (16,) * spatial_dims  # type: ignore
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))  # type: ignore
        self.hidden_size = hidden_size
        self.classification = False
        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            classification=self.classification,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
            qkv_bias=qkv_bias,
        )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)
        self.proj_axes = (0, spatial_dims + 1) + tuple(d + 1 for d in range(spatial_dims))
        self.proj_view_shape = list(self.feat_size) + [self.hidden_size]

    def proj_feat(self, x: torch.Tensor) -> torch.Tensor:
        """Project the feature map to the hidden size."""
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:  # noqa: D102
        """Forward function for the network."""
        x, hidden_states_out = self.vit(x_in)
        enc1 = self.encoder1(x_in)
        x2 = hidden_states_out[3]
        enc2 = self.encoder2(self.proj_feat(x2))
        x3 = hidden_states_out[6]
        enc3 = self.encoder3(self.proj_feat(x3))
        x4 = hidden_states_out[9]
        enc4 = self.encoder4(self.proj_feat(x4))
        dec4 = self.proj_feat(x)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)
        return self.out(out)
