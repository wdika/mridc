# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from typing import List, Sequence, Union

import numpy as np
import torch
from omegaconf import ListConfig

__all__ = ["fft2", "ifft2", "fftshift", "ifftshift"]


def fft2(
    data: torch.Tensor,
    centered: bool = False,
    normalization: str = "backward",
    spatial_dims: Sequence[int] = None,
) -> torch.Tensor:
    """
    Apply 2 dimensional Fast Fourier Transform.

    Parameters
    ----------
    data : Complex valued input data containing at least 2 dimensions: dimensions -2 & -1 are spatial dimensions.
    centered : Whether to center the fft. If True, the fft will be shifted so that the zero frequency component is
        in the center of the spectrum. Default is False.
    normalization : Normalization mode. For the forward transform (fft2()), these correspond to:
        * "forward" - normalize by 1/n
        * "backward" - no normalization
        * "ortho" - normalize by 1/sqrt(n) (making the FFT orthonormal)
        Where n = prod(s) is the logical FFT size.
        Calling the backward transform (ifft2()) with the same normalization mode will apply an overall normalization
        of 1/n between the two transforms. This is required to make ifft2() the exact inverse.
        Default is "backward" (no normalization).
    spatial_dims : Dimensions to apply the FFT. Default is the last two dimensions.
        If tensor is viewed as real, the last dimension is assumed to be the complex dimension.

    Returns
    -------
    The 2D FFT of the input.

    Examples
    --------
    >>> import torch
    >>> from mridc.collections.common.parts.fft import fft2
    >>> x = torch.randn(2, 3, 4, 5, 2)
    >>> fft2(x).shape
    torch.Size([2, 3, 4, 5, 2])
    >>> fft2(x, centered=True).shape
    torch.Size([2, 3, 4, 5, 2])
    >>> fft2(x, centered=True, normalization="ortho").shape
    torch.Size([2, 3, 4, 5, 2])
    >>> fft2(x, centered=True, normalization="ortho", spatial_dims=[-3, -2]).shape
    torch.Size([2, 3, 4, 5, 2])

    Notes
    -----
    The PyTorch fft2 function does not support complex tensors. Therefore, the input is converted to a complex tensor
    and then converted back to a real tensor. This is done by using the torch.view_as_complex and torch.view_as_real
    functions. The input is assumed to be a real tensor with the last dimension being the complex dimension.

    The PyTorch fft2 function performs a separate fft, so fft2 is the same as fft(fft(data, dim=-2), dim=-1).

    Source: https://pytorch.org/docs/stable/fft.html#torch.fft.fft2
    """
    if data.shape[-1] == 2:
        data = torch.view_as_complex(data)

    if spatial_dims is None:
        spatial_dims = [-2, -1]
    elif isinstance(spatial_dims, ListConfig):
        spatial_dims = list(spatial_dims)

    if centered:
        data = ifftshift(data, dim=spatial_dims)

    data = torch.fft.fft2(
        data,
        dim=spatial_dims,
        norm=normalization if normalization.lower() != "none" else None,
    )

    if centered:
        data = fftshift(data, dim=spatial_dims)

    data = torch.view_as_real(data)

    return data


def ifft2(
    data: torch.Tensor,
    centered: bool = False,
    normalization: str = "backward",
    spatial_dims: Sequence[int] = None,
) -> torch.Tensor:
    """
    Apply 2 dimensional Inverse Fast Fourier Transform.

    Parameters
    ----------
    data : Complex valued input data containing at least 2 dimensions: dimensions -2 & -1 are spatial dimensions.
    centered : Whether to center the ifft. If True, the ifft will be shifted so that the zero frequency component is
        in the center of the spectrum. Default is False.
    normalization : Normalization mode. For the backward transform (ifft2()), these correspond to:
        * "forward" - normalize by 1/n
        * "backward" - no normalization
        * "ortho" - normalize by 1/sqrt(n) (making the IFFT orthonormal)
        Where n = prod(s) is the logical IFFT size.
        Calling the forward transform (fft2()) with the same normalization mode will apply an overall normalization
        of 1/n between the two transforms. This is required to make ifft2() the exact inverse.
        Default is "backward" (no normalization).
    spatial_dims : Dimensions to apply the IFFT. Default is the last two dimensions.
        If tensor is viewed as real, the last dimension is assumed to be the complex dimension.

    Returns
    -------
    The 2D IFFT of the input.

    Examples
    --------
    >>> import torch
    >>> from mridc.collections.common.parts.fft import ifft2
    >>> x = torch.randn(2, 3, 4, 5, 2)
    >>> ifft2(x).shape
    torch.Size([2, 3, 4, 5, 2])
    >>> ifft2(x, centered=True).shape
    torch.Size([2, 3, 4, 5, 2])
    >>> ifft2(x, centered=True, normalization="ortho").shape
    torch.Size([2, 3, 4, 5, 2])
    >>> ifft2(x, centered=True, normalization="ortho", spatial_dims=[-3, -2]).shape
    torch.Size([2, 3, 4, 5, 2])

    Notes
    -----
    The PyTorch ifft2 function does not support complex tensors. Therefore, the input is converted to a complex tensor
    and then converted back to a real tensor. This is done by using the torch.view_as_complex and torch.view_as_real
    functions. The input is assumed to be a real tensor with the last dimension being the complex dimension.

    The PyTorch ifft2 function performs a separate ifft, so ifft2 is the same as ifft(ifft(data, dim=-2), dim=-1).

    Source: https://pytorch.org/docs/stable/fft.html#torch.fft.ifft2
    """
    if data.shape[-1] == 2:
        data = torch.view_as_complex(data)

    if spatial_dims is None:
        spatial_dims = [-2, -1]
    elif isinstance(spatial_dims, ListConfig):
        spatial_dims = list(spatial_dims)

    if centered:
        data = ifftshift(data, dim=spatial_dims)

    data = torch.fft.ifft2(
        data,
        dim=spatial_dims,
        norm=normalization if normalization.lower() != "none" else None,
    )

    if centered:
        data = fftshift(data, dim=spatial_dims)

    data = torch.view_as_real(data)

    return data


def roll_one_dim(data: torch.Tensor, shift: int, dim: int) -> torch.Tensor:
    """
    Similar to roll but for only one dim.

    Parameters
    ----------
    data : A PyTorch tensor.
    shift : Amount to roll.
    dim : Which dimension to roll.

    Returns
    -------
    Rolled version of data.

    Examples
    --------
    >>> import torch
    >>> from mridc.collections.common.parts.fft import roll_one_dim
    >>> x = torch.randn(2, 3, 4, 5)
    >>> roll_one_dim(x, 1, 0).shape
    torch.Size([2, 3, 4, 5])

    Notes
    -----
    Source: https://github.com/facebookresearch/fastMRI/blob/main/fastmri/fftc.py
    """
    shift %= data.size(dim)
    if shift == 0:
        return data

    left = data.narrow(dim, 0, data.size(dim) - shift)
    right = data.narrow(dim, data.size(dim) - shift, shift)

    return torch.cat((right, left), dim=dim)


def roll(data: torch.Tensor, shift: List[int], dim: Union[List[int], Sequence[int]]) -> torch.Tensor:
    """
    Similar to np.roll but applies to PyTorch Tensors.

    Parameters
    ----------
    data : A PyTorch tensor.
    shift : Amount to roll.
    dim : Which dimension to roll.

    Returns
    -------
    Rolled version of data.

    Examples
    --------
    >>> import torch
    >>> from mridc.collections.common.parts.fft import roll
    >>> x = torch.randn(2, 3, 4, 5)
    >>> roll(x, [1, 2], [0, 1]).shape
    torch.Size([2, 3, 4, 5])

    Notes
    -----
    Source: https://github.com/facebookresearch/fastMRI/blob/main/fastmri/fftc.py
    """
    if len(shift) != len(dim):
        raise ValueError("len(shift) must match len(dim)")

    if isinstance(dim, ListConfig):
        dim = list(dim)

    for (s, d) in zip(shift, dim):
        data = roll_one_dim(data, s, d)

    return data


def fftshift(data: torch.Tensor, dim: Union[List[int], Sequence[int]] = None) -> torch.Tensor:
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors

    Parameters
    ----------
    data : A PyTorch tensor.
    dim : Which dimension to fftshift.

    Returns
    -------
    fftshifted version of data.

    Examples
    --------
    >>> import torch
    >>> from mridc.collections.common.parts.fft import fftshift
    >>> x = torch.randn(2, 3, 4, 5)
    >>> fftshift(x).shape
    torch.Size([2, 3, 4, 5])

    Notes
    -----
    Source: https://github.com/facebookresearch/fastMRI/blob/main/fastmri/fftc.py
    """
    if dim is None:
        # this weird code is necessary for torch.jit.script typing
        dim = [0] * (data.dim())
        for i in range(1, data.dim()):
            dim[i] = i
    elif isinstance(dim, ListConfig):
        dim = list(dim)

    # Also necessary for torch.jit.script
    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = np.floor_divide(data.shape[dim_num], 2)

    return roll(data, shift, dim)


def ifftshift(data: torch.Tensor, dim: Union[List[int], Sequence[int]] = None) -> torch.Tensor:
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors

    Parameters
    ----------
    data : A PyTorch tensor.
    dim : Which dimension to ifftshift.

    Returns
    -------
    ifftshifted version of data.

    Examples
    --------
    >>> import torch
    >>> from mridc.collections.common.parts.fft import ifftshift
    >>> x = torch.randn(2, 3, 4, 5)
    >>> ifftshift(x).shape
    torch.Size([2, 3, 4, 5])

    Notes
    -----
    Source: https://github.com/facebookresearch/fastMRI/blob/main/fastmri/fftc.py
    """
    if dim is None:
        # this weird code is necessary for torch.jit.script typing
        dim = [0] * (data.dim())
        for i in range(1, data.dim()):
            dim[i] = i
    elif isinstance(dim, ListConfig):
        dim = list(dim)

    # Also necessary for torch.jit.script
    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = np.floor_divide(data.shape[dim_num] + 1, 2)

    return roll(data, shift, dim)
