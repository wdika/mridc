# coding=utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from:
# https://github.com/Project-MONAI/MONAI/blob/c38d503a587f1779914bd071a1b2d66a6d9080c2/monai/networks/layers/weight_init.py#L45

import math
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn


class Convolution(nn.Sequential):
    """
    Constructs a convolution with normalization, optional dropout, and optional activation layers::
        -- (Conv|ConvTrans) -- (Norm -- Dropout -- Acti) --
    if ``conv_only`` set to ``True``::
        -- (Conv|ConvTrans) --

    Parameters
    ----------
    spatial_dims : int
        Number of spatial dimensions.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : Union[Sequence[int], int]
        Size of the convolving kernel.
    stride : Union[Sequence[int], int], optional
        Stride of the convolution. Default is ``1``.
    kernel_size : Union[Sequence[int], int]
        Size of the convolving kernel. Default is ``3``.
    adn_ordering : str, optional
        A string representing the ordering of activation, normalization, and dropout. Default is ``"NDA"``.
    act : Union[Type[nn.Module], Tuple[Type[nn.Module], dict]], optional
        Activation type and arguments. Default is ``PReLU``.
    norm : Union[Type[nn.Module], Tuple[Type[nn.Module], dict]], optional
        Feature normalization type and arguments. Default is ``instance norm``.
    dropout : float, optional
        Dropout ratio. Default is ``no dropout``.
    dropout_dim : int, optional
        Determine the spatial dimensions of dropout. Default is ``1``. \
        - When dropout_dim = 1, randomly zeroes some of the elements for each channel.
        - When dropout_dim = 2, Randomly zeroes out entire channels (a channel is a 2D feature map).
        - When dropout_dim = 3, Randomly zeroes out entire channels (a channel is a 3D feature map).
        The value of dropout_dim should be no larger than the value of `spatial_dims`.
    dilation : int, optional
        Dilation rate. Default is ``1``.
    groups : int, optional
        Controls the connections between inputs and outputs. Default is ``1``.
    bias : bool, optional
        Whether to have a bias term. Default is ``True``.
    conv_only : bool, optional
        Whether to use the convolutional layer only. Default is ``False``.
    is_transposed : bool, optional
        If ``True`` uses ConvTrans instead of Conv. Default is ``False``.
    padding : Union[Sequence[int], int], optional
        Controls the amount of implicit zero-paddings on both sides for padding number of points for each dimension.
        Default is ``None``.
    output_padding : Union[Sequence[int], int], optional
        Controls the additional size added to one side of the output shape. Default is ``None``.

    .. note::
        This is a wrapper for monai implementation. See:
        https://github.com/Project-MONAI/MONAI/blob/c38d503a587f1779914bd071a1b2d66a6d9080c2/monai/networks/layers/\
        weight_init.py
    """

    def __init__(  # noqa: C901
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        strides: Union[Sequence[int], int] = 1,
        kernel_size: Union[Sequence[int], int] = 3,
        adn_ordering: str = "NDA",  # noqa: A003
        act: Optional[Union[Tuple, str]] = "PRELU",  # noqa: A003
        norm: Optional[Union[Tuple, str]] = "INSTANCE",  # noqa: A003
        dropout: Optional[Union[Tuple, str, float]] = None,  # noqa: A003
        dropout_dim: Optional[int] = 1,  # noqa: A003
        dilation: Union[Sequence[int], int] = 1,
        groups: int = 1,
        bias: bool = True,
        conv_only: bool = False,  # noqa: A003
        is_transposed: bool = False,
        padding: Optional[Union[Sequence[int], int]] = None,
        output_padding: Optional[Union[Sequence[int], int]] = None,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_transposed = is_transposed
        if padding is None:
            padding = same_padding(kernel_size, dilation)

        if self.spatial_dims == 1:
            if is_transposed:
                conv_type = nn.ConvTranspose1d
            else:
                conv_type = nn.Conv1d
        elif self.spatial_dims == 2:
            if is_transposed:
                conv_type = nn.ConvTranspose2d
            else:
                conv_type = nn.Conv2d
        elif self.spatial_dims == 3:
            if is_transposed:
                conv_type = nn.ConvTranspose3d
            else:
                conv_type = nn.Conv3d
        else:
            raise ValueError(f"Unsupported spatial_dims: {self.spatial_dims}")

        conv: nn.Module
        if is_transposed:
            if output_padding is None:
                output_padding = stride_minus_kernel_padding(1, strides)
            conv = conv_type(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=strides,
                padding=padding,
                output_padding=output_padding,
                groups=groups,
                bias=bias,
                dilation=dilation,
            )
        else:
            conv = conv_type(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=strides,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )

        self.add_module("conv", conv)


def stride_minus_kernel_padding(
    kernel_size: Union[Sequence[int], int], stride: Union[Sequence[int], int]
) -> Union[Tuple[int, ...], int]:
    """
    Calculate the output padding for the given kernel size and stride.

    Parameters
    ----------
    kernel_size : Union[Sequence[int], int]
        The kernel size.
    stride : Union[Sequence[int], int]
        The stride.

    Returns
    -------
    Union[Tuple[int, ...], int]
        The output padding.
    """
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)

    out_padding_np = stride_np - kernel_size_np
    out_padding = tuple(int(p) for p in out_padding_np)

    return out_padding if len(out_padding) > 1 else out_padding[0]


def get_padding(
    kernel_size: Union[Sequence[int], int], stride: Union[Sequence[int], int]
) -> Union[Tuple[int, ...], int]:
    """
    Calculate the padding for the given kernel size and stride.

    Parameters
    ----------
    kernel_size : Union[Sequence[int], int]
        The kernel size.
    stride : Union[Sequence[int], int]
        The stride.

    Returns
    -------
    Union[Tuple[int, ...], int]
        The padding.
    """
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = (kernel_size_np - stride_np + 1) / 2
    if np.min(padding_np) < 0:
        raise AssertionError("padding value should not be negative, please change the kernel size and/or stride.")
    padding = tuple(int(p) for p in padding_np)

    return padding if len(padding) > 1 else padding[0]


def get_output_padding(
    kernel_size: Union[Sequence[int], int], stride: Union[Sequence[int], int], padding: Union[Sequence[int], int]
) -> Union[Tuple[int, ...], int]:
    """
    Calculate the output padding for the given kernel size, stride and padding.

    Parameters
    ----------
    kernel_size : Union[Sequence[int], int]
        The kernel size.
    stride : Union[Sequence[int], int]
        The stride.

    Returns
    -------
    Union[Tuple[int, ...], int]
        The output padding.
    """
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = np.atleast_1d(padding)

    out_padding_np = 2 * padding_np + stride_np - kernel_size_np
    if np.min(out_padding_np) < 0:
        raise AssertionError("out_padding value should not be negative, please change the kernel size and/or stride.")
    out_padding = tuple(int(p) for p in out_padding_np)

    return out_padding if len(out_padding) > 1 else out_padding[0]


def same_padding(
    kernel_size: Union[Sequence[int], int], dilation: Union[Sequence[int], int] = 1
) -> Union[Tuple[int, ...], int]:
    """
    Return the padding value needed to ensure a convolution using the given kernel size produces an output of the same
    shape as the input for a stride of 1, otherwise ensure a shape of the input divided by the stride rounded down.

    Raises
    ------
        NotImplementedError: When ``np.any((kernel_size - 1) * dilation % 2 == 1)``.

    Parameters
    ----------
    kernel_size : Union[Sequence[int], int]
        The kernel size.
    dilation : Union[Sequence[int], int]
        The dilation.

    Returns
    -------
    Union[Tuple[int, ...], int]
        The padding.
    """
    kernel_size_np = np.atleast_1d(kernel_size)
    dilation_np = np.atleast_1d(dilation)

    if np.any((kernel_size_np - 1) * dilation % 2 == 1):
        raise NotImplementedError(
            f"Same padding not available for kernel_size={kernel_size_np} and dilation={dilation_np}."
        )

    padding_np = (kernel_size_np - 1) / 2 * dilation_np
    padding = tuple(int(p) for p in padding_np)

    return padding if len(padding) > 1 else padding[0]


def get_conv_layer(  # noqa: C901
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size: Union[Sequence[int], int] = 3,
    stride: Union[Sequence[int], int] = 1,
    act: Optional[Union[Tuple, str]] = nn.PReLU,
    norm: Optional[Union[Tuple, str]] = nn.InstanceNorm2d,
    dropout: Optional[Union[Tuple, str, float]] = None,
    bias: bool = False,
    conv_only: bool = True,
    is_transposed: bool = False,
) -> Convolution:
    """
    Get a convolution layer with the given parameters.

    Parameters
    ----------
    spatial_dims : int
        The number of spatial dimensions.
    in_channels : int
        The number of input channels.
    out_channels : int
        The number of output channels.
    kernel_size : Union[Sequence[int], int]
        The kernel size. Default is ``3``.
    stride : Union[Sequence[int], int]
        The stride. Default is ``1``.
    act : Optional[Union[Tuple, str]]
        The activation function. Default is ``nn.PReLU``.
    norm : Optional[Union[Tuple, str]]
        The normalization layer. Default is ``nn.InstanceNorm2d``.
    dropout : Optional[Union[Tuple, str, float]]
        The dropout layer. Default is ``None``.
    bias : bool
        Whether to add a bias. Default is ``False``.
    conv_only : bool
        Whether to only return the convolution layer. Default is ``True``.
    is_transposed : bool
        Whether to use a transposed convolution. Default is ``False``.

    Returns
    -------
    Convolution
        The convolution layer.
    """
    padding = get_padding(kernel_size, stride)
    output_padding = None
    if is_transposed:
        output_padding = get_output_padding(kernel_size, stride, padding)
    return Convolution(
        spatial_dims,
        in_channels,
        out_channels,
        strides=stride,
        kernel_size=kernel_size,
        act=act,
        norm=norm,
        dropout=dropout,
        bias=bias,
        conv_only=conv_only,
        is_transposed=is_transposed,
        padding=padding,
        output_padding=output_padding,
    )


def _no_grad_trunc_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0):
    """
    Tensor initialization with truncated normal distribution.

    Based on:
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    https://github.com/rwightman/pytorch-image-models

    Parameters
    ----------
    tensor : torch.Tensor
        The tensor to initialize.
    mean : float
        The mean of the normal distribution. Default is ``0.0``.
    std : float
        The standard deviation of the normal distribution. Default is ``1.0``.
    a : float
        The lower bound of the truncated normal distribution. Default is ``-2.0``.
    b : float
        The upper bound of the truncated normal distribution. Default is ``2.0``.
    """

    def norm_cdf(x: float) -> float:
        """Normal cumulative distribution function."""
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        tensor.uniform_(2 * norm_cdf((a - mean) / std) - 1, 2 * norm_cdf((b - mean) / std) - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(
    tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0
) -> torch.Tensor:
    """
    Tensor initialization with truncated normal distribution.

    Based on:
    https://github.com/rwightman/pytorch-image-models

    Parameters
    ----------
    tensor : torch.Tensor
        The tensor to initialize.
    mean : float
        The mean of the normal distribution. Default is ``0.0``.
    std : float
        The standard deviation of the normal distribution. Default is ``1.0``.
    a : float
        The lower bound of the truncated normal distribution. Default is ``-2.0``.
    b : float
        The upper bound of the truncated normal distribution. Default is ``2.0``.
    """
    if std <= 0:
        raise ValueError("the standard deviation should be greater than zero.")
    if a >= b:
        raise ValueError("minimum cutoff value (a) should be smaller than maximum cutoff value (b).")
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
