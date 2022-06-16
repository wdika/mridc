import torch
from torch import nn

from mridc.collections.common.parts.fft import fft2, ifft2


def mse_gradient(x, data, fft_centered, fft_normalization, spatial_dims):
    """
    Calculates the gradient under a linear forward model

    x: image estimate
    data: [y,mask], y - zero-filled image reconstruction, mask - sub-sampling mask
    fft_centered: Whether to center the fft.
    fft_normalization: "ortho" is the default normalization used by PyTorch. Can be changed to "ortho" or None.
    spatial_dims: dimensions to apply the FFT
    """
    y, mask = data[0], data[1]
    x = torch.chunk(x, 2, 1)
    x = torch.view_as_real(torch.complex(x[0], x[1]))
    x = fft2(x, fft_centered, fft_normalization, spatial_dims)
    x = mask * x
    x = ifft2(x, fft_centered, fft_normalization, spatial_dims)
    x = torch.cat((x[..., 0], x[..., 1]), 1)
    return x


def determine_conv_class(n_dim, transposed=False):
    if n_dim == 1:
        return nn.Conv1d if not transposed else nn.ConvTranspose1d
    elif n_dim == 2:
        return nn.Conv2d if not transposed else nn.ConvTranspose2d
    elif n_dim == 3:
        return nn.Conv3d if not transposed else nn.ConvTranspose3d
    else:
        NotImplementedError("No convolution of this dimensionality implemented")


def determine_conv_functional(n_dim, transposed=False):
    if n_dim == 1:
        if not transposed:
            return nn.functional.conv1d
        else:
            return nn.functional.conv_transposed1d
    elif n_dim == 2:
        if not transposed:
            return nn.functional.conv2d
        else:
            return nn.functional.conv_transposed2d
    elif n_dim == 3:
        if not transposed:
            return nn.functional.conv3d
        else:
            return nn.functional.conv_transposed3d
    else:
        NotImplementedError("No convolution of this dimensionality implemented")


def pixel_unshuffle(x, downscale_factor):
    b, c, h, w = x.size()
    r = downscale_factor
    out_channel = c * (r**2)
    out_h = h // r
    out_w = w // r
    x_view = x.contiguous().view(b, c, out_h, r, out_w, r)
    return x_view.permute(0, 1, 3, 5, 2, 4).contiguous().view(b, out_channel, out_h, out_w)


class PixelUnshuffle(nn.Module):
    def __init__(self, downscale_factor):
        super(PixelUnshuffle, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, x):
        return pixel_unshuffle(x, self.downscale_factor)
