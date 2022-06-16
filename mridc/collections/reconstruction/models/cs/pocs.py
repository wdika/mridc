# coding=utf-8
__author__ = "Dimitrios Karkalousos"

import ptwt
import pywt
import torch

from mridc.collections.common.parts.fft import fft2, ifft2


def POCS(y, mask, fft_centered, fft_normalization, spatial_dims, soft_threshold_val, max_iter):
    """
    Projection Over a Convex Set algorithm.

    Args:
        y: input image.
        mask: binary mask.
        fft_centered: Whether to center the fft.
        fft_normalization: "ortho" is the default normalization used by PyTorch. Can be changed to "ortho" or None.
        spatial_dims: dimensions to apply the FFT
        soft_threshold_val: soft threshold value.
        max_iter: maximum number of iterations.

    Returns:
        l: reconstructed image.
    """
    soft_threshold_val = torch.nn.Parameter(torch.tensor(soft_threshold_val))

    lines = ifft2(y, fft_centered, fft_normalization, spatial_dims)
    for _ in range(max_iter):
        wavelet_l = ptwt.wavedec2(lines, pywt.Wavelet("db4"))
        wavelet_l = FullSoftThresholding_wavedec(wavelet_l, soft_threshold_val)
        lines = ptwt.waverec2(wavelet_l, pywt.Wavelet("db4"))
        lines = ifft2(
            fft2(lines, fft_centered, fft_normalization, spatial_dims) * (1 - mask) + y,
            fft_centered,
            fft_normalization,
            spatial_dims,
        )
    return lines


def SoftThresholding(lines, soft_threshold_val):
    """Soft thresholding function."""
    zero = torch.zeros_like(lines)
    return torch.where(
        torch.abs(lines) > soft_threshold_val, (torch.abs(lines) - soft_threshold_val) * lines / torch.abs(lines), zero
    )


def FullSoftThresholding_wavedec(lines, soft_threshold_val):
    """Full Soft thresholding function."""
    res = [SoftThresholding(lines[0], soft_threshold_val)]
    res.extend(
        (
            SoftThresholding(e[0], soft_threshold_val),
            SoftThresholding(e[1], soft_threshold_val),
            SoftThresholding(e[2], soft_threshold_val),
        )
        for e in lines[1:]
    )

    return res
