# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Parts of the code have been taken from https://github.com/facebookresearch/fastMRI

from typing import List, Optional, Union

import numpy as np
import torch

__all__ = ["fft2c", "ifft2c"]


def fft2c(
    data: torch.Tensor,
    fft_type: str = "orthogonal",
    fft_normalization: str = "ortho",
    fft_dim: Union[Optional[int], List[int], None] = None,
) -> torch.Tensor:
    """
    Apply centered 2 dimensional Fast Fourier Transform.

    Parameters
    ----------
    data: Complex valued input data containing at least 3 dimensions: dimensions -2 & -1 are spatial dimensions. All
    other dimensions are assumed to be batch dimensions.
    fft_type: Specify fft type. This is important if an orthogonal transformation is needed or not.
    fft_normalization: "ortho" is the default normalization used by PyTorch. Can be changed to "ortho" or None.
    fft_dim: dimensions to apply the FFT

    Returns
    -------
    The FFT of the input.
    """
    if fft_dim is None:
        fft_dim = [-2, -1]

    if fft_type == "orthogonal":
        data = ifftshift(data, dim=[-3, -2])

    data = torch.view_as_real(torch.fft.fft2(torch.view_as_complex(data), dim=fft_dim, norm=fft_normalization))

    if fft_type == "orthogonal":
        data = fftshift(data, dim=[-3, -2])

    return data


def ifft2c(
    data: torch.Tensor,
    fft_type: str = "orthogonal",
    fft_normalization: str = "ortho",
    fft_dim: Union[Optional[int], List[int], None] = None,
) -> torch.Tensor:
    """
    Apply centered 2 dimensional Inverse Fast Fourier Transform.

    Parameters
    ----------
    data: Complex valued input data containing at least 3 dimensions: dimensions -2 & -1 are spatial dimensions. All
    other dimensions are assumed to be batch dimensions.
    fft_type: Specify fft type. This is important if an orthogonal transformation is needed or not.
    fft_normalization: "ortho" is the default normalization used by PyTorch. Can be changed to "ortho" or None.
    fft_dim: dimensions to apply the FFT

    Returns
    -------
    The IFFT of the input.
    """
    if fft_dim is None:
        fft_dim = [-2, -1]

    if fft_type == "orthogonal":
        data = ifftshift(data, dim=[-3, -2])

    data = torch.view_as_real(torch.fft.ifft2(torch.view_as_complex(data), dim=fft_dim, norm=fft_normalization))

    if fft_type == "orthogonal":
        data = fftshift(data, dim=[-3, -2])

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


def roll(x: torch.Tensor, shift: List[int], dim: List[int]) -> torch.Tensor:
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

    for (s, d) in zip(shift, dim):
        x = roll_one_dim(x, s, d)

    return x


def fftshift(x: torch.Tensor, dim: Optional[List[int]] = None) -> torch.Tensor:
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

    # Also necessary for torch.jit.script
    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = np.floor_divide(x.shape[dim_num], 2)

    return roll(x, shift, dim)


def ifftshift(x: torch.Tensor, dim: Optional[List[int]] = None) -> torch.Tensor:
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

    # Also necessary for torch.jit.script
    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = np.floor_divide(x.shape[dim_num] + 1, 2)

    return roll(x, shift, dim)
