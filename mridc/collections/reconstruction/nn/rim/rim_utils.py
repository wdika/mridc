# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from typing import Sequence

import torch

from mridc.collections.common.parts import fft


def log_likelihood_gradient(  # noqa: W0221
    prediction: torch.Tensor,
    masked_kspace: torch.Tensor,
    sensitivity_maps: torch.Tensor,
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
    prediction : torch.Tensor
        Initial guess for the reconstruction. Shape [batch_size, height, width, 2].
    masked_kspace : torch.Tensor
        Subsampled k-space data. Shape [batch_size, coils, height, width, 2].
    sensitivity_maps : torch.Tensor
        Coil sensitivity maps. Shape [batch_size, coils, height, width, 2].
    mask : torch.Tensor
        Subsampling mask. Shape [batch_size, 1, height, width, 1].
    sigma : float
        Noise level.
    fft_centered : bool
        Whether to center the FFT.
    fft_normalization : str
        Whether to normalize the FFT.
    spatial_dims : Sequence[int]
        Spatial dimensions of the data.
    coil_dim : int
        Dimension of the coil.

    Returns
    -------
    torch.Tensor
        Gradient of the log-likelihood function. Shape [batch_size, 4, height, width]. 4 is the stacked real and
        imaginary parts of the prediction and the real and imaginary parts of the gradient.
    """
    if coil_dim == 0:
        coil_dim += 1

    prediction_real, prediction_imag = map(lambda x: torch.unsqueeze(x, coil_dim), prediction.chunk(2, -1))
    sensitivity_maps_real, sensitivity_maps_imag = sensitivity_maps.chunk(2, -1)

    re_se = prediction_real * sensitivity_maps_real - prediction_imag * sensitivity_maps_imag
    im_se = prediction_real * sensitivity_maps_imag + prediction_imag * sensitivity_maps_real
    pred = torch.cat((re_se, im_se), -1)

    pred = fft.fft2(pred, centered=fft_centered, normalization=fft_normalization, spatial_dims=spatial_dims)

    pred = fft.ifft2(
        mask * (pred - masked_kspace),
        centered=fft_centered,
        normalization=fft_normalization,
        spatial_dims=spatial_dims,
    )
    pred_real, pred_imag = pred.chunk(2, -1)

    re_out = torch.sum(pred_real * sensitivity_maps_real + pred_imag * sensitivity_maps_imag, coil_dim) / (sigma**2.0)
    im_out = torch.sum(pred_imag * sensitivity_maps_real - pred_real * sensitivity_maps_imag, coil_dim) / (sigma**2.0)

    prediction_real = prediction_real.squeeze(coil_dim)
    prediction_imag = prediction_imag.squeeze(coil_dim)

    return torch.cat((prediction_real, prediction_imag, re_out, im_out), -1).permute(0, 3, 1, 2)
