# coding=utf-8
__author__ = "Dimitrios Karkalousos"

import ptwt
import pywt
import torch

from mridc.collections.common.parts.fft import fft2, ifft2
from mridc.collections.common.parts.utils import complex_conj


def objective_function(m, y, lines, mask, fft_centered, fft_normalization, spatial_dims):
    """
    Objective function.

    Parameters
    ----------
    m : torch.Tensor
        Input data.
    y : torch.Tensor
        Reference input data.
    lines : torch.Tensor
        Initial guess of the lines.
    mask : torch.Tensor
        Mask of the data.
    fft_centered : bool
        Whether to center the FFT.
    fft_normalization : str
        Type of FFT normalization.
    spatial_dims : tuple
        Spatial dimensions of the data.

    Returns
    -------
    Objective function value.
    """
    wm = ptwt.wavedec2(m, pywt.Wavelet("db4"))
    l1norm = torch.sum(torch.abs(wm[0]))
    for e in wm[1:]:
        for i in range(3):
            l1norm += torch.sum(torch.abs(e[i]))
    return torch.linalg.norm(fft2(m, fft_centered, fft_normalization, spatial_dims) * mask - y) ** 2 + lines * l1norm


def grad_objective_function(m, y, lines, mask, mu, fft_centered, fft_normalization, spatial_dims):
    """
    Gradient of the objective function.

    Parameters
    ----------
    m : torch.Tensor
        Input data.
    y : torch.Tensor
        Reference input data.
    lines : torch.Tensor
        Initial guess of the lines.
    mask : torch.Tensor
        Mask of the data.
    mu : float
        Soft threshold value.
    fft_centered : bool
        Whether to center the FFT.
    fft_normalization : str
        Type of FFT normalization.
    spatial_dims : tuple
        Spatial dimensions of the data.

    Returns
    -------
    Gradient of the objective function.
    """
    wm = ptwt.wavedec2(m, pywt.Wavelet("db4"))
    wm[0] = (1 / torch.sqrt(wm[0] ** 2 + mu)) * wm[0]
    # for i in range(1, 7):
    #     detail_list = [(1 / torch.sqrt(wm[i][j] ** 2 + mu)) * wm[i][j] for j in range(3)]
    #     wm[i] = tuple(detail_list)
    return 2 * ifft2(
        fft2(m, fft_centered, fft_normalization, spatial_dims) * mask - y,
        fft_centered,
        fft_normalization,
        spatial_dims,
    ) + lines * ptwt.waverec2(wm, pywt.Wavelet("db4"))


def conjugate_gradient(y, mask, lines, max_iter, mu, alpha, beta, fft_centered, fft_normalization, spatial_dims):
    """
    Conjugate Gradient algorithm.

    Parameters
    ----------
    y : torch.Tensor
        Input data.
    mask : torch.Tensor
        Mask of the data.
    lines : torch.Tensor
        Initial guess of the lines.
    max_iter : int
        Maximum number of iterations.
    mu : float
        Soft threshold value.
    alpha : float
        Step size.
    beta : float
        Step size.
    fft_centered : bool
        Whether to center the FFT.
    fft_normalization : str
        Type of FFT normalization.
    spatial_dims : tuple
        Spatial dimensions of the data.

    Returns
    -------
    Reconstructed data.
    """
    m = torch.zeros_like(y)
    g = grad_objective_function(m, y, lines, mask, mu, fft_centered, fft_normalization, spatial_dims)
    delta_m = -g
    m_list = []
    for _ in range(max_iter):
        # backtracking line-search
        t = 1
        obj_m = objective_function(m, y, lines, mask, fft_centered, fft_normalization, spatial_dims)
        while objective_function(
            m + t * delta_m, y, lines, mask, fft_centered, fft_normalization, spatial_dims
        ) > obj_m + alpha * t * torch.real(torch.sum(complex_conj(g) * delta_m)):
            t *= beta

        # step
        m = m + t * delta_m
        m = torch.abs(m)

        # update gradients and momentum
        g_1 = g.clone()
        g = grad_objective_function(m, y, lines, mask, mu, fft_centered, fft_normalization, spatial_dims)
        gamma = (torch.linalg.norm(g) / torch.linalg.norm(g_1)) ** 2
        delta_m = -g + gamma * delta_m

        # append result
        m_list.append(m)

    return m_list
