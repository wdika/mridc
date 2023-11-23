# coding=utf-8
import time
from typing import Tuple

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import binary_dilation
from skimage.filters import threshold_otsu
from skimage.morphology import convex_hull_image
from tqdm import tqdm

from mridc.collections.quantitative.parts.transforms import R2star_B0_S0_phi_mapping, R2star_mapping

# data = h5py.File("/data/projects/recon/data/public/ahead_preprocessing/mp2rageme_001_axial.h5", "r")
# print(data.keys())
#
# slice_idx = 0
#
# kspace = data["kspace"][()]
# sensitivity_map = data["sensitivity_map"][()]
#
# kspace = kspace[slice_idx]
# sensitivity_map = sensitivity_map[slice_idx]
#
# imspace = np.fft.ifftn(kspace, axes=(-2, -1))
# axial_target = np.sum(imspace * np.conj(sensitivity_map), axis=1)
#
# # axial_target = data["target"][()]
# axial_target = torch.from_numpy(axial_target)
# axial_target = torch.view_as_real(axial_target)
# # axial_target = axial_target[slice_idx]
#
# print(axial_target.shape)
# head_mask = data["mask_head"][()]
# head_mask = torch.from_numpy(head_mask)
# head_mask = head_mask[slice_idx]
# brain_mask = data["mask_brain"][()]
# brain_mask = torch.from_numpy(brain_mask)
# brain_mask = brain_mask[slice_idx]
#
# plt.subplot(1, 4, 1)
# plt.imshow(torch.abs(torch.view_as_complex(axial_target[0])), cmap="gray")
# plt.subplot(1, 4, 2)
# plt.imshow(torch.angle(torch.view_as_complex(axial_target[0])), cmap="gray")
# plt.subplot(1, 4, 3)
# plt.imshow(torch.real(torch.view_as_complex(axial_target[0])), cmap="gray")
# plt.subplot(1, 4, 4)
# plt.imshow(torch.imag(torch.view_as_complex(axial_target[0])), cmap="gray")
# plt.show()


axial_target = h5py.File(
    "/data/projects/recon/other/dkarkalousos/MultiTaskLearning_rebuttal/mridc/axial_target.h5", "r"
)["target"][()]
axial_target = torch.from_numpy(axial_target)
axial_target = torch.view_as_real(axial_target)
axial_target = axial_target[100]

# plt.subplot(1, 4, 1)
# plt.imshow(torch.abs(torch.view_as_complex(axial_target[0])), cmap="gray")
# plt.subplot(1, 4, 2)
# plt.imshow(torch.angle(torch.view_as_complex(axial_target[0])), cmap="gray")
# plt.subplot(1, 4, 3)
# plt.imshow(torch.real(torch.view_as_complex(axial_target[0])), cmap="gray")
# plt.subplot(1, 4, 4)
# plt.imshow(torch.imag(torch.view_as_complex(axial_target[0])), cmap="gray")
# plt.show()

axial_target_np = torch.abs(torch.view_as_complex(axial_target)).numpy()
head_mask = np.abs(axial_target_np[0, :, :]) > threshold_otsu(np.abs(axial_target_np[0, :, :]))
head_mask = binary_dilation(head_mask, iterations=4)
head_mask = convex_hull_image(head_mask).astype(np.float32)
head_mask = torch.from_numpy(head_mask)
brain_mask = 1 - head_mask

TEs = [3.0, 11.5, 20.0, 28.5]  # four echo times


start = time.time()
R2star_map, S0_map_real, B0_map, S0_map_imag = R2star_B0_S0_phi_mapping(
    axial_target, TEs, brain_mask, head_mask, spatial_dims=[-2, -1]
)
end = time.time()
print(end - start)

plt.subplot(2, 2, 1)
plt.imshow(R2star_map, cmap="gray")
plt.subplot(2, 2, 2)
plt.imshow(S0_map_real, cmap="gray")
plt.subplot(2, 2, 3)
plt.imshow(B0_map, cmap="gray")
plt.subplot(2, 2, 4)
plt.imshow(S0_map_imag, cmap="gray")
plt.show()
