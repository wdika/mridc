# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Parts of the code have been taken from https://github.com/facebookresearch/fastMRI

import numpy as np
import pytest
import torch

from mridc.collections.common.parts.fft import fft2, fftshift, ifft2, ifftshift, roll
from mridc.collections.common.parts.utils import complex_abs, tensor_to_complex_np
from tests.collections.reconstruction.fastmri.conftest import create_input


@pytest.mark.parametrize("shape", [[3, 3], [4, 6], [10, 8, 4]])
def test_centered_fft2(shape):
    """
    Test centered 2D Fast Fourier Transform.

    Args:
        shape: shape of the input

    Returns:
        None
    """
    shape = shape + [2]
    x = create_input(shape)
    out_torch = fft2(x, centered=True, normalization="ortho", spatial_dims=[-2, -1]).numpy()
    out_torch = out_torch[..., 0] + 1j * out_torch[..., 1]

    input_numpy = tensor_to_complex_np(x)
    input_numpy = np.fft.ifftshift(input_numpy, (-2, -1))
    out_numpy = np.fft.fft2(input_numpy, norm="ortho")
    out_numpy = np.fft.fftshift(out_numpy, (-2, -1))

    if not np.allclose(out_torch, out_numpy):
        raise AssertionError


@pytest.mark.parametrize("shape", [[3, 3], [4, 6], [10, 8, 4]])
def test_non_centered_fft2(shape):
    """
    Test non-centered 2D Fast Fourier Transform.

    Args:
        shape: shape of the input

    Returns:
        None
    """
    shape = shape + [2]
    x = create_input(shape)
    out_torch = fft2(x, centered=False, normalization="ortho", spatial_dims=[-2, -1]).numpy()
    out_torch = out_torch[..., 0] + 1j * out_torch[..., 1]

    input_numpy = tensor_to_complex_np(x)
    out_numpy = np.fft.fft2(input_numpy, norm="ortho")

    if not np.allclose(out_torch, out_numpy):
        raise AssertionError


@pytest.mark.parametrize("shape", [[3, 3], [4, 6], [10, 8, 4]])
def test_centered_fft2_backward_normalization(shape):
    """
    Test centered 2D Fast Fourier Transform with backward normalization.

    Args:
        shape: shape of the input

    Returns:
        None
    """
    shape = shape + [2]
    x = create_input(shape)
    out_torch = fft2(x, centered=True, normalization="backward", spatial_dims=[-2, -1]).numpy()
    out_torch = out_torch[..., 0] + 1j * out_torch[..., 1]

    input_numpy = tensor_to_complex_np(x)
    input_numpy = np.fft.ifftshift(input_numpy, (-2, -1))
    out_numpy = np.fft.fft2(input_numpy, norm="backward")
    out_numpy = np.fft.fftshift(out_numpy, (-2, -1))

    if not np.allclose(out_torch, out_numpy):
        raise AssertionError


@pytest.mark.parametrize("shape", [[3, 3], [4, 6], [10, 8, 4]])
def test_centered_fft2_forward_normalization(shape):
    """
    Test centered 2D Fast Fourier Transform with forward normalization.

    Args:
        shape: shape of the input

    Returns:
        None
    """
    shape = shape + [2]
    x = create_input(shape)
    out_torch = fft2(x, centered=True, normalization="forward", spatial_dims=[-2, -1]).numpy()
    out_torch = out_torch[..., 0] + 1j * out_torch[..., 1]

    input_numpy = tensor_to_complex_np(x)
    input_numpy = np.fft.ifftshift(input_numpy, (-2, -1))
    out_numpy = np.fft.fft2(input_numpy, norm="forward")
    out_numpy = np.fft.fftshift(out_numpy, (-2, -1))

    if not np.allclose(out_torch, out_numpy):
        raise AssertionError


@pytest.mark.parametrize("shape", [[3, 3], [4, 6], [10, 8, 4]])
def test_centered_ifft2(shape):
    """
    Test centered 2D Inverse Fast Fourier Transform.

    Args:
        shape: shape of the input

    Returns:
        None
    """
    shape = shape + [2]
    x = create_input(shape)
    out_torch = ifft2(x, centered=True, normalization="ortho", spatial_dims=[-2, -1]).numpy()
    out_torch = out_torch[..., 0] + 1j * out_torch[..., 1]

    input_numpy = tensor_to_complex_np(x)
    input_numpy = np.fft.ifftshift(input_numpy, (-2, -1))
    out_numpy = np.fft.ifft2(input_numpy, norm="ortho")
    out_numpy = np.fft.fftshift(out_numpy, (-2, -1))

    if not np.allclose(out_torch, out_numpy):
        raise AssertionError


@pytest.mark.parametrize("shape", [[3, 3], [4, 6], [10, 8, 4]])
def test_non_centered_ifft2(shape):
    """
    Test non-centered 2D Inverse Fast Fourier Transform.

    Args:
        shape: shape of the input

    Returns:
        None
    """
    shape = shape + [2]
    x = create_input(shape)
    out_torch = ifft2(x, centered=False, normalization="ortho", spatial_dims=[-2, -1]).numpy()
    out_torch = out_torch[..., 0] + 1j * out_torch[..., 1]

    input_numpy = tensor_to_complex_np(x)
    out_numpy = np.fft.ifft2(input_numpy, norm="ortho")

    if not np.allclose(out_torch, out_numpy):
        raise AssertionError


@pytest.mark.parametrize("shape", [[3, 3], [4, 6], [10, 8, 4]])
def test_centered_ifft2_backward_normalization(shape):
    """
    Test centered 2D Inverse Fast Fourier Transform with backward normalization.

    Args:
        shape: shape of the input

    Returns:
        None
    """
    shape = shape + [2]
    x = create_input(shape)
    out_torch = ifft2(x, centered=True, normalization="backward", spatial_dims=[-2, -1]).numpy()
    out_torch = out_torch[..., 0] + 1j * out_torch[..., 1]

    input_numpy = tensor_to_complex_np(x)
    input_numpy = np.fft.ifftshift(input_numpy, (-2, -1))
    out_numpy = np.fft.ifft2(input_numpy, norm="backward")
    out_numpy = np.fft.fftshift(out_numpy, (-2, -1))

    if not np.allclose(out_torch, out_numpy):
        raise AssertionError


@pytest.mark.parametrize("shape", [[3, 3], [4, 6], [10, 8, 4]])
def test_centered_ifft2_forward_normalization(shape):
    """
    Test centered 2D Inverse Fast Fourier Transform with forward normalization.

    Args:
        shape: shape of the input

    Returns:
        None
    """
    shape = shape + [2]
    x = create_input(shape)
    out_torch = ifft2(x, centered=True, normalization="forward", spatial_dims=[-2, -1]).numpy()
    out_torch = out_torch[..., 0] + 1j * out_torch[..., 1]

    input_numpy = tensor_to_complex_np(x)
    input_numpy = np.fft.ifftshift(input_numpy, (-2, -1))
    out_numpy = np.fft.ifft2(input_numpy, norm="forward")
    out_numpy = np.fft.fftshift(out_numpy, (-2, -1))

    if not np.allclose(out_torch, out_numpy):
        raise AssertionError


@pytest.mark.parametrize("shape", [[3, 3], [4, 6], [10, 8, 4]])
def test_complex_abs(shape):
    """
    Test complex absolute value.

    Args:
        shape: shape of the input

    Returns:
        None
    """
    shape = shape + [2]
    x = create_input(shape)
    out_torch = complex_abs(x).numpy()
    input_numpy = tensor_to_complex_np(x)
    out_numpy = np.abs(input_numpy)

    if not np.allclose(out_torch, out_numpy):
        raise AssertionError


@pytest.mark.parametrize("shift, dim", [(0, 0), (1, 0), (-1, 0), (100, 0), ([1, 2], [1, 2])])
@pytest.mark.parametrize("shape", [[5, 6, 2], [3, 4, 5]])
def test_roll(shift, dim, shape):
    """
    Test roll.

    Args:
        shift: shift of the input
        dim: dimension of the input
        shape: shape of the input

    Returns:
        None
    """
    x = np.arange(np.product(shape)).reshape(shape)
    if isinstance(shift, int) and isinstance(dim, int):
        torch_shift = [shift]
        torch_dim = [dim]
    else:
        torch_shift = shift
        torch_dim = dim
    out_torch = roll(torch.from_numpy(x), torch_shift, torch_dim).numpy()
    out_numpy = np.roll(x, shift, dim)

    if not np.allclose(out_torch, out_numpy):
        raise AssertionError


@pytest.mark.parametrize("shape", [[5, 3], [2, 4, 6]])
def test_fftshift(shape):
    """
    Test fftshift.

    Args:
        shape: shape of the input

    Returns:
        None
    """
    x = np.arange(np.product(shape)).reshape(shape)
    out_torch = fftshift(torch.from_numpy(x)).numpy()
    out_numpy = np.fft.fftshift(x)

    if not np.allclose(out_torch, out_numpy):
        raise AssertionError


@pytest.mark.parametrize("shape", [[5, 3], [2, 4, 5], [2, 7, 5]])
def test_ifftshift(shape):
    """
    Test ifftshift.

    Args:
        shape: shape of the input

    Returns:
        None
    """
    x = np.arange(np.product(shape)).reshape(shape)
    out_torch = ifftshift(torch.from_numpy(x)).numpy()
    out_numpy = np.fft.ifftshift(x)

    if not np.allclose(out_torch, out_numpy):
        raise AssertionError
