# coding=utf-8
__author__ = "Dimitrios Karkalousos"

import torch

from mridc.collections.common.parts.fft import fft2c, ifft2c


def log_likelihood_gradient(
    eta: torch.Tensor,
    masked_kspace: torch.Tensor,
    sense: torch.Tensor,
    mask: torch.Tensor,
    sigma: float,
    fft_type: str = "orthogonal",
) -> torch.Tensor:
    """
    Computes the gradient of the log-likelihood function.

    Args:
        eta: Initial guess for the reconstruction.
        masked_kspace: Masked k-space data.
        sense: Sensing matrix.
        mask: Mask.
        sigma: Noise level.
        fft_type: Type of FFT to use.

    Returns
    -------
    torch.Tensor: Gradient of the log-likelihood function.
    """
    eta_real, eta_imag = map(lambda x: torch.unsqueeze(x, 0), eta.chunk(2, -1))
    sense_real, sense_imag = sense.chunk(2, -1)

    re_se = eta_real * sense_real - eta_imag * sense_imag
    im_se = eta_real * sense_imag + eta_imag * sense_real

    pred = ifft2c(mask * (fft2c(torch.cat((re_se, im_se), -1), fft_type=fft_type) - masked_kspace), fft_type=fft_type)

    pred_real, pred_imag = pred.chunk(2, -1)

    re_out = torch.sum(pred_real * sense_real + pred_imag * sense_imag, 1) / (sigma ** 2.0)
    im_out = torch.sum(pred_imag * sense_real - pred_real * sense_imag, 1) / (sigma ** 2.0)

    eta_real = eta_real.squeeze(0)
    eta_imag = eta_imag.squeeze(0)

    return torch.cat((eta_real, eta_imag, re_out, im_out), 0).unsqueeze(0).squeeze(-1)
