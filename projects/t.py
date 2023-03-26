# coding=utf-8
# Parts of the code have been taken from https://github.com/facebookresearch/fastMRI
__author__ = "Dimitrios Karkalousos"

import contextlib
from typing import Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import torch
from matplotlib import pyplot as plt

from mridc.collections.common.data.subsample import create_masker
from mridc.collections.common.parts.fft import ifft2
from mridc.collections.common.parts.utils import apply_mask, to_tensor

if __name__ == "__main__":
    coil_dim = 1
    fft_centered = False
    fft_normalization = "backward"
    spatial_dims = (-2, -1)

    slice = 30

    accelerations = [3]
    center_fractions = [0.7]
    shift = False

    data = "/data/projects/recon/data/private/3T_T1_3D_Brains/h5/multicoil_test/119_transversal"

    with h5py.File(data, "r") as f:
        kspace = to_tensor(h5py.File(data)["kspace"][()])[slice]

    masked_kspace, mask, acc = apply_mask(
        kspace,
        create_masker("gaussian1d", center_fractions, accelerations),
        None,
        None,
        shift=shift,
        half_scan_percentage=0,
        center_scale=0.02,
    )
    mask = np.repeat(mask, masked_kspace.shape[1], axis=1)
    mask = np.abs(mask).squeeze().byte().numpy()

    print(mask.size / mask.sum())

    target = torch.abs(
        torch.view_as_complex(
            torch.sqrt(torch.sum(ifft2(kspace, fft_centered, fft_normalization, spatial_dims) ** 2, dim=coil_dim - 1))
        )
    )
    masked_target = torch.abs(
        torch.view_as_complex(
            torch.sqrt(
                torch.sum(ifft2(masked_kspace, fft_centered, fft_normalization, spatial_dims) ** 2, dim=coil_dim - 1)
            )
        )
    )

    plt.figure(figsize=(10, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(target, cmap="gray")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(masked_target, cmap="gray")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(np.fft.ifftshift(mask), cmap="gray")
    plt.axis("off")
    # plt.subplot(1, 5, 4)
    # plt.imshow(masked_target1d, cmap="gray")
    # plt.axis("off")
    # plt.subplot(1, 5, 5)
    # plt.imshow(np.fft.ifftshift(mask1d), cmap="gray")
    # plt.axis("off")
    plt.show()
