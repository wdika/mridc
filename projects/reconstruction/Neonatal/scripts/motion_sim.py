"""make circle and perform motion simulation"""

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from mridc.collections.common.parts.random_motion import MotionSimulation
from mridc.collections.reconstruction.data.subsample import create_mask_for_mask_type
from mridc.collections.reconstruction.parts.utils import apply_mask
from mridc.collections.reconstruction.parts.transforms import NoisePreWhitening
from mridc.collections.common.parts.utils import to_tensor, complex_mul, complex_conj
from mridc.collections.common.parts.fft import fft2, ifft2


def plot_im(im, title=None, *args, **kwargs):
    plt.imshow(im, cmap='gray', *args, **kwargs), plt.axis('off')
    if title:
        plt.title(title)


if __name__ == '__main__':
    data = h5py.File('/data/projects/recon/data/private/3T_T1_3D_Brains/h5/multicoil_val/109_transversal', 'r')

    slice = 10
    coil_dim = 1

    fft_centered = False
    fft_normalization = 'backward'
    spatial_dims = (-2, -1)

    device = 'cuda'

    kspace = to_tensor(data['kspace'][()][slice]).to(device)
    sensitivity_map = to_tensor(data['sensitivity_map'][()][slice]).to(device)

    kspace, _, _ = apply_mask(
        kspace,
        create_mask_for_mask_type('poisson2d', [.0], [14]),
        None,
        None,
        shift=True,
        half_scan_percentage=0,
        center_scale=0.02,
    )

    imspace = ifft2(kspace, centered=fft_centered, normalization=fft_normalization, spatial_dims=spatial_dims)

    ms_layer = MotionSimulation(
        type='gaussian',
        angle=3,
        translation=3,
        center_percentage=0.0,
        motion_percentage=[100, 100],
        spatial_dims=spatial_dims,
        num_segments=64,
        random_num_segments=False,
        non_uniform=False,
    )

    exp_phase_shift = ms_layer.forward(kspace)
    motion_kspace = torch.view_as_real(torch.multiply(exp_phase_shift, torch.view_as_complex(kspace).flatten()).reshape(kspace.shape[:-1]))
    motion_imspace = ifft2(motion_kspace, centered=fft_centered, normalization=fft_normalization, spatial_dims=spatial_dims)

    translations = ms_layer.params['translations'].T.reshape(kspace.shape[:-1] + (3,))
    # translations_target = torch.sum(translations, coil_dim-1).cpu()
    # # translations_sq = torch.sum(translations_target ** 2, dim=-1)
    # plt.subplot(1, 3, 1)
    # plt.imshow(translations_target[..., 0].cpu().numpy(), cmap='gray')
    # plt.subplot(1, 3, 2)
    # plt.imshow(translations_target[..., 1].cpu().numpy(), cmap='gray')
    # plt.subplot(1, 3, 3)
    # plt.imshow(translations_target[..., 2].cpu().numpy(), cmap='gray')
    # plt.show()

    prewhitening_patch_start = 10
    prewhitening_patch_length = 10
    prewhitening_scale_factor = 1.0
    prewhitening = NoisePreWhitening(
        patch_size=[
        prewhitening_patch_start,
        prewhitening_patch_length + prewhitening_patch_start,
        prewhitening_patch_start,
        prewhitening_patch_length + prewhitening_patch_start,
        ],
        scale_factor=prewhitening_scale_factor,
    )
    pw_motion_kspace = prewhitening(motion_kspace)
    pw_motion_imspace = ifft2(pw_motion_kspace, centered=fft_centered, normalization=fft_normalization, spatial_dims=spatial_dims)

    target = torch.view_as_complex(torch.sum(complex_mul(imspace, complex_conj(sensitivity_map)), coil_dim-1)).cpu()
    motion_target = torch.view_as_complex(torch.sum(complex_mul(motion_imspace, complex_conj(sensitivity_map)), coil_dim-1)).cpu()
    pw_motion_target = torch.view_as_complex(torch.sum(complex_mul(pw_motion_imspace, complex_conj(sensitivity_map)), coil_dim-1)).cpu()

    # plt.figure()
    # plt.subplot(3, 4, 1)
    # plot_im(np.abs(target), title='original magnitude')
    # plt.subplot(3, 4, 2)
    # plot_im(np.angle(target), title='original phase')
    # plt.subplot(3, 4, 3)
    # plot_im(np.real(target), title='original real')
    # plt.subplot(3, 4, 4)
    # plot_im(np.imag(target), title='original imaginary')
    # plt.subplot(3, 4, 5)
    # plot_im(np.abs(motion_target), title='motion-simulation magnitude')
    # plt.subplot(3, 4, 6)
    # plot_im(np.angle(motion_target), title='motion-simulation phase')
    # plt.subplot(3, 4, 7)
    # plot_im(np.real(motion_target), title='motion-simulation real')
    # plt.subplot(3, 4, 8)
    # plot_im(np.imag(motion_target), title='motion-simulation imaginary')
    # plt.subplot(3, 4, 9)
    # plot_im(np.abs(pw_motion_target), title='prewhitened motion-simulation magnitude')
    # plt.subplot(3, 4, 10)
    # plot_im(np.angle(pw_motion_target), title='prewhitened motion-simulation phase')
    # plt.subplot(3, 4, 11)
    # plot_im(np.real(pw_motion_target), title='prewhitened motion-simulation real')
    # plt.subplot(3, 4, 12)
    # plot_im(np.imag(pw_motion_target), title='prewhitened motion-simulation imaginary')
    # plt.show()
