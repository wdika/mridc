# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Parts of the code have been taken from https://github.com/facebookresearch/fastMRI

from typing import List, Sequence, Union

import numpy as np
import torch
from omegaconf import ListConfig

__all__ = ["fft2", "ifft2"]


def fft2(
    data: torch.Tensor,
    centered: bool = True,
    normalization: str = "ortho",
    spatial_dims: Sequence[int] = None,
) -> torch.Tensor:
    """
    Apply 2 dimensional Fast Fourier Transform.

    Parameters
    ----------
    data: Complex valued input data containing at least 3 dimensions: dimensions -2 & -1 are spatial dimensions. All
    other dimensions are assumed to be batch dimensions.
    centered: Whether to center the fft.
    normalization: "ortho" is the default normalization used by PyTorch. Can be changed to "ortho" or None.
    spatial_dims: dimensions to apply the FFT

    Returns
    -------
    The FFT of the input.
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
    centered: bool = True,
    normalization: str = "ortho",
    spatial_dims: Sequence[int] = None,
) -> torch.Tensor:
    """
    Apply 2 dimensional Inverse Fast Fourier Transform.

    Parameters
    ----------
    data: Complex valued input data containing at least 3 dimensions: dimensions -2 & -1 are spatial dimensions. All
    other dimensions are assumed to be batch dimensions.
    centered: Whether to center the fft.
    normalization: "ortho" is the default normalization used by PyTorch. Can be changed to "ortho" or None.
    spatial_dims: dimensions to apply the FFT

    Returns
    -------
    The FFT of the input.
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


def roll_one_dim(x: torch.Tensor, shift: int, dim: int) -> torch.Tensor:
    """
    Similar to roll but for only one dim.

    Parameters
    ----------
    x: A PyTorch tensor.
    shift: Amount to roll.
    dim: Which dimension to roll.

    Returns
    -------
    Rolled version of x.
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
    x: A PyTorch tensor.
    shift: Amount to roll.
    dim: Which dimension to roll.

    Returns
    -------
    Rolled version of x.
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
    Similar to np.fft.fftshift but applies to PyTorch Tensors

    Parameters
    ----------
    x: A PyTorch tensor.
    dim: Which dimension to fftshift.

    Returns
    -------
    fftshifted version of x.
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
    Similar to np.fft.ifftshift but applies to PyTorch Tensors

    Parameters
    ----------
    x: A PyTorch tensor.
    dim: Which dimension to ifftshift.

    Returns
    -------
    ifftshifted version of x.
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
