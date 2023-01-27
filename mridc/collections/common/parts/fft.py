# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from typing import List, Sequence, Union

import numpy as np
import torch
from omegaconf import ListConfig

__all__ = ["fft2", "ifft2", "fftshift", "ifftshift"]

"""
This module contains custom and wrapper functions for performing Fast Fourier Transform (FFT) and Inverse Fast Fourier
Transform (IFFT) on PyTorch tensors. The functions are designed to work with complex valued tensors.
"""


def fft2(
    x: torch.Tensor,
    centered: bool = False,
    normalization: str = "backward",
    spatial_dims: Sequence[int] = None,
) -> torch.Tensor:
    """
    Apply 2 dimensional Fast Fourier Transform.

    Parameters
    ----------
    x : torch.Tensor
        Complex valued input data, where dimensions -2 & -1 are spatial dimensions.
    centered : bool
        Whether to center the fft. If True, the fft will be shifted so that the zero frequency component is in the
        center of the spectrum. Default is ``False``.
    normalization : str
        Normalization mode. For the forward transform (fft2()), these correspond to: \n
            * ``forward`` - normalize by 1/n
            * ``backward`` - no normalization
            * ``ortho`` - normalize by 1/sqrt(n) (making the FFT orthonormal)
        Where n = prod(s) is the logical FFT size.
        Calling the backward transform (ifft2()) with the same normalization mode will apply an overall
        normalization of 1/n between the two transforms. This is required to make ifft2() the exact inverse.
        Default is ``backward`` (no normalization).
    spatial_dims : Sequence[int]
        Dimensions to apply the FFT. Default is the last two dimensions. If tensor is viewed as real, the last
        dimension is assumed to be the complex dimension.

    Returns
    -------
    torch.Tensor
        The 2D FFT of the input.

    Examples
    --------
    >>> import torch
    >>> from mridc.collections.common.parts.fft import fft2
    >>> data = torch.randn(2, 3, 4, 5, 2)
    >>> fft2(data).shape
    torch.Size([2, 3, 4, 5, 2])
    >>> fft2(data, centered=True, normalization="ortho", spatial_dims=[-3, -2]).shape
    torch.Size([2, 3, 4, 5, 2])

    .. note::
        The PyTorch fft2 function does not support complex tensors. Therefore, the input is converted to a complex
        tensor and then converted back to a real tensor. This is done by using the torch.view_as_complex and
        torch.view_as_real functions. The input is assumed to be a real tensor with the last dimension being the
        complex dimension.

        The PyTorch fft2 function performs a separate fft, so fft2 is the same as fft(fft(data, dim=-2), dim=-1).

        Source: https://pytorch.org/docs/stable/fft.html#torch.fft.fft2
    """
    if x.shape[-1] == 2:
        x = torch.view_as_complex(x)

    if spatial_dims is None:
        spatial_dims = [-2, -1]
    elif isinstance(spatial_dims, ListConfig):
        spatial_dims = list(spatial_dims)

    if centered:
        x = ifftshift(x, dim=spatial_dims)

    x = torch.fft.fft2(
        x,
        dim=spatial_dims,
        norm=normalization if normalization.lower() != "none" else None,
    )

    if centered:
        x = fftshift(x, dim=spatial_dims)

    x = torch.view_as_real(x)

    return x


def ifft2(
    x: torch.Tensor,
    centered: bool = False,
    normalization: str = "backward",
    spatial_dims: Sequence[int] = None,
) -> torch.Tensor:
    """
    Apply 2 dimensional Inverse Fast Fourier Transform.

    Parameters
    ----------
    x : torch.Tensor
        Complex valued input data, where dimensions -2 & -1 are spatial dimensions.
    centered : bool
        Whether to center the ifft. If True, the ifft will be shifted so that the zero frequency component is in the
        center of the spectrum. Default is ``False``.
    normalization : str
        Normalization mode. For the backward transform (ifft2()), these correspond to: \n
            * ``forward`` - normalize by 1/n
            * ``backward`` - no normalization
            * ``ortho`` - normalize by 1/sqrt(n) (making the IFFT orthonormal)
        Where n = prod(s) is the logical IFFT size.
        Calling the forward transform (fft2()) with the same normalization mode will apply an overall
        normalization of 1/n between the two transforms. This is required to make fft2() the exact inverse.
        Default is ``backward`` (no normalization).
    spatial_dims : Sequence[int]
        Dimensions to apply the IFFT. Default is the last two dimensions. If tensor is viewed as real, the last
        dimension is assumed to be the complex dimension.

    Returns
    -------
    torch.Tensor
        The 2D IFFT of the input.

    Examples
    --------
    >>> import torch
    >>> from mridc.collections.common.parts.fft import ifft2
    >>> data = torch.randn(2, 3, 4, 5, 2)
    >>> ifft2(data).shape
    torch.Size([2, 3, 4, 5, 2])
    >>> ifft2(data, centered=True, normalization="ortho", spatial_dims=[-3, -2]).shape
    torch.Size([2, 3, 4, 5, 2])

    .. note::
        The PyTorch ifft2 function does not support complex tensors. Therefore, the input is converted to a complex
        tensor and then converted back to a real tensor. This is done by using the torch.view_as_complex and
        torch.view_as_real functions. The input is assumed to be a real tensor with the last dimension being the
        complex dimension.

        The PyTorch ifft2 function performs a separate ifft, so ifft2 is the same as ifft(ifft(data, dim=-2), dim=-1).

        Source: https://pytorch.org/docs/stable/fft.html#torch.fft.ifft2
    """
    if x.shape[-1] == 2:
        x = torch.view_as_complex(x)

    if spatial_dims is None:
        spatial_dims = [-2, -1]
    elif isinstance(spatial_dims, ListConfig):
        spatial_dims = list(spatial_dims)

    if centered:
        x = ifftshift(x, dim=spatial_dims)

    x = torch.fft.ifft2(
        x,
        dim=spatial_dims,
        norm=normalization if normalization.lower() != "none" else None,
    )

    if centered:
        x = fftshift(x, dim=spatial_dims)

    x = torch.view_as_real(x)

    return x


def roll_one_dim(x: torch.Tensor, shift: int, dim: int) -> torch.Tensor:
    """
    Similar to roll but for only one dim.

    Parameters
    ----------
    x : torch.Tensor
        Input data.
    shift : int
        Amount to roll.
    dim : int
        Which dimension to roll.

    Returns
    -------
    torch.Tensor
        The rolled tensor.

    Examples
    --------
    >>> import torch
    >>> from mridc.collections.common.parts.fft import roll_one_dim
    >>> data = torch.randn(2, 3, 4, 5)
    >>> roll_one_dim(data, 1, 0).shape
    torch.Size([2, 3, 4, 5])

    .. note::
        Source: https://github.com/facebookresearch/fastMRI/blob/main/fastmri/fftc.py
    """
    shift %= x.size(dim)
    if shift == 0:
        return x

    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)

    return torch.cat((right, left), dim=dim)


def roll(x: torch.Tensor, shift: List[int], dim: Union[List[int], Sequence[int]]) -> torch.Tensor:
    """
    Similar to np.roll but applies to PyTorch Tensors.

    Parameters
    ----------
    x : torch.Tensor
        Input data.
    shift : List[int]
        Amount to roll.
    dim : Union[List[int], Sequence[int]]
        Which dimension to roll.

    Returns
    -------
    torch.Tensor
        The rolled tensor.

    Examples
    --------
    >>> import torch
    >>> from mridc.collections.common.parts.fft import roll
    >>> data = torch.randn(2, 3, 4, 5)
    >>> roll(data, [1, 2], [0, 1]).shape
    torch.Size([2, 3, 4, 5])

    .. note::
        Source: https://github.com/facebookresearch/fastMRI/blob/main/fastmri/fftc.py
    """
    if len(shift) != len(dim):
        raise ValueError("len(shift) must match len(dim)")

    if isinstance(dim, ListConfig):
        dim = list(dim)

    for (s, d) in zip(shift, dim):
        x = roll_one_dim(x, s, d)

    return x


def fftshift(x: torch.Tensor, dim: Union[List[int], Sequence[int]] = None) -> torch.Tensor:
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors.

    Parameters
    ----------
    x : torch.Tensor
        Input data.
    dim : Union[List[int], Sequence[int]]
        Which dimension to shift.

    Returns
    -------
    torch.Tensor
        The shifted tensor.

    Examples
    --------
    >>> import torch
    >>> from mridc.collections.common.parts.fft import fftshift
    >>> data = torch.randn(2, 3, 4, 5)
    >>> fftshift(data).shape
    torch.Size([2, 3, 4, 5])

    .. note::
        Source: https://github.com/facebookresearch/fastMRI/blob/main/fastmri/fftc.py
    """
    if dim is None:
        # this weird code is necessary for torch.jit.script typing
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i
    elif isinstance(dim, ListConfig):
        dim = list(dim)

    # Also necessary for torch.jit.script
    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = np.floor_divide(x.shape[dim_num], 2)

    return roll(x, shift, dim)


def ifftshift(x: torch.Tensor, dim: Union[List[int], Sequence[int]] = None) -> torch.Tensor:
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors.

    Parameters
    ----------
    x : torch.Tensor
        Input data.
    dim : Union[List[int], Sequence[int]]
        Which dimension to shift.

    Returns
    -------
    torch.Tensor
        The shifted tensor.

    Examples
    --------
    >>> import torch
    >>> from mridc.collections.common.parts.fft import ifftshift
    >>> data = torch.randn(2, 3, 4, 5)
    >>> ifftshift(data).shape
    torch.Size([2, 3, 4, 5])

    .. note::
        Source: https://github.com/facebookresearch/fastMRI/blob/main/fastmri/fftc.py
    """
    if dim is None:
        # this weird code is necessary for torch.jit.script typing
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i
    elif isinstance(dim, ListConfig):
        dim = list(dim)

    # Also necessary for torch.jit.script
    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = np.floor_divide(x.shape[dim_num] + 1, 2)

    return roll(x, shift, dim)
