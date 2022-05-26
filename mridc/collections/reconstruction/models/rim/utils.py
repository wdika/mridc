# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from typing import Sequence

import torch

from mridc.collections.common.parts.fft import fft2, ifft2


def log_likelihood_gradient(
    eta: torch.Tensor,
    masked_kspace: torch.Tensor,
    sense: torch.Tensor,
    mask: torch.Tensor,
    sigma: float,
    fft_centered: bool,
    fft_normalization: str,
    spatial_dims: Sequence[int],
    coil_dim: int,
) -> torch.Tensor:
    """
    Computes the gradient of the log-likelihood function.

    Parameters
    ----------
    eta: Initial guess for the reconstruction.
    masked_kspace: Subsampled k-space data.
    sense: Sensing matrix.
    mask: Sampling mask.
    sigma: Noise level.
    fft_centered: Whether to center the FFT.
    fft_normalization: Whether to normalize the FFT.
    spatial_dims: Spatial dimensions of the data.
    coil_dim: Dimension of the coil.

    Returns
    -------
    Gradient of the log-likelihood function.
    """
    coil_dim = 1

    eta_real, eta_imag = map(lambda x: torch.unsqueeze(x, coil_dim), eta.chunk(2, -1))
    sense_real, sense_imag = sense.chunk(2, -1)

    re_se = eta_real * sense_real - eta_imag * sense_imag
    im_se = eta_real * sense_imag + eta_imag * sense_real
    pred = torch.cat((re_se, im_se), -1)

    pred = fft2(pred, centered=fft_centered, normalization=fft_normalization, spatial_dims=spatial_dims)

    pred = ifft2(
        mask * (pred - masked_kspace),
        centered=fft_centered,
        normalization=fft_normalization,
        spatial_dims=spatial_dims,
    )
    pred_real, pred_imag = pred.chunk(2, -1)

    re_out = torch.sum(pred_real * sense_real + pred_imag * sense_imag, coil_dim) / (sigma**2.0)
    im_out = torch.sum(pred_imag * sense_real - pred_real * sense_imag, coil_dim) / (sigma**2.0)

    eta_real = eta_real.squeeze(coil_dim)
    eta_imag = eta_imag.squeeze(coil_dim)

    return torch.cat((eta_real, eta_imag, re_out, im_out), -1).permute(0, 3, 1, 2)
