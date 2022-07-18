"""make circle and perform motion simulation"""

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

from mridc.collections.common.parts.fft import ifft2
from mridc.collections.common.parts.utils import complex_conj, complex_mul, to_tensor
from mridc.collections.motioncompensation.parts.deformation import generate_affine_grid, generate_nonlinear_grid


def plot_im(im, title=None, *args, **kwargs):
    plt.imshow(im, cmap="gray", *args, **kwargs), plt.axis("off")
    if title:
        plt.title(title)


if __name__ == "__main__":
    data = h5py.File("/data/projects/recon/data/private/3T_T1_3D_Brains/h5/multicoil_val/109_transversal", "r")

    slice = 50
    coil_dim = 1

    fft_centered = False
    fft_normalization = "backward"
    spatial_dims = (-2, -1)

    device = "cuda"

    kspace = to_tensor(data["kspace"][()][slice]).to(device)
    sensitivity_map = to_tensor(data["sensitivity_map"][()][slice]).to(device)

    # kspace, _, _ = apply_mask(
    #     kspace,
    #     create_mask_for_mask_type('poisson2d', [.0], [14]),
    #     None,
    #     None,
    #     shift=True,
    #     half_scan_percentage=0,
    #     center_scale=0.02,
    # )

    imspace = ifft2(kspace, centered=fft_centered, normalization=fft_normalization, spatial_dims=spatial_dims)

    translation = (0, 0)
    rotation = (10, 0)
    scale = 0
    P = 0
    theta = 0
    sigma = 0
    align_corners = False

    index_slice = 50

    affine_grid = generate_affine_grid(
        shape=imspace[:1, :, :, 0].shape,
        translation=[
            2 * translation[0] * np.random.rand(1) - translation[1],
            2 * translation[0] * np.random.rand(1) - translation[1],
        ],
        rotate=2 * rotation[0] * np.random.rand(1) - rotation[1],
        scale=2 * scale * np.random.rand(1) - scale + 1,
    ).to(imspace)
    non_linear_grid = generate_nonlinear_grid(shape=imspace[:1, :, :, 0].shape, P=P, theta=theta, sigma=sigma).to(
        imspace
    )

    # imspace = torch.sum(complex_mul(imspace, complex_conj(sensitivity_map)), coil_dim-1).unsqueeze(coil_dim-1)

    m_imspace_real = torch.nn.functional.grid_sample(
        imspace[..., 0].unsqueeze(0), affine_grid, mode="bilinear", align_corners=align_corners
    )
    m_imspace_real = torch.nn.functional.grid_sample(
        m_imspace_real, non_linear_grid, mode="bilinear", align_corners=align_corners
    )

    m_imspace_imag = torch.nn.functional.grid_sample(
        imspace[..., 1].unsqueeze(0), affine_grid, mode="bilinear", align_corners=align_corners
    )
    m_imspace_imag = torch.nn.functional.grid_sample(
        m_imspace_imag, non_linear_grid, mode="bilinear", align_corners=align_corners
    )

    motion_imspace = torch.cat([m_imspace_real, m_imspace_imag], 0).permute(1, 2, 3, 0)
    motion_imspace = motion_imspace[..., 0] + 1j * motion_imspace[..., 1]
    motion_imspace = torch.view_as_real(motion_imspace).squeeze(0)

    target = torch.view_as_complex(torch.sum(complex_mul(imspace, complex_conj(sensitivity_map)), coil_dim - 1)).cpu()
    motion_target = torch.view_as_complex(
        torch.sum(complex_mul(motion_imspace, complex_conj(sensitivity_map)), coil_dim - 1)
    ).cpu()

    plt.figure()
    plt.subplot(3, 4, 1)
    plot_im(np.abs(target), title="original magnitude")
    plt.subplot(3, 4, 2)
    plot_im(np.angle(target), title="original phase")
    plt.subplot(3, 4, 3)
    plot_im(np.real(target), title="original real")
    plt.subplot(3, 4, 4)
    plot_im(np.imag(target), title="original imaginary")
    plt.subplot(3, 4, 5)
    plot_im(np.abs(motion_target), title="motion-simulation magnitude")
    plt.subplot(3, 4, 6)
    plot_im(np.angle(motion_target), title="motion-simulation phase")
    plt.subplot(3, 4, 7)
    plot_im(np.real(motion_target), title="motion-simulation real")
    plt.subplot(3, 4, 8)
    plot_im(np.imag(motion_target), title="motion-simulation imaginary")
    plt.subplot(3, 4, 9)
    plot_im(np.abs(target - motion_target), title="motion-simulation magnitude")
    plt.subplot(3, 4, 10)
    plot_im(np.angle(target - motion_target), title="motion-simulation phase")
    plt.subplot(3, 4, 11)
    plot_im(np.real(target - motion_target), title="motion-simulation real")
    plt.subplot(3, 4, 12)
    plot_im(np.imag(target - motion_target), title="motion-simulation imaginary")
    plt.show()
