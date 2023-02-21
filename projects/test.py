# coding=utf-8
# import matplotlib.pyplot as plt
# import torch
# from omegaconf import DictConfig
#
# import mridc
#
# x = torch.randn(15, 256, 256)
# x = x + 1j * torch.randn(15, 256, 256)
# x = torch.view_as_real(x)
# print(f"x: {x.shape}")
#
# assert not mridc.is_none(x)
# y = mridc.fft.fft2(x)
# assert not mridc.is_none(y)
# pw = mridc.transforms.GeometricDecompositionCoilCompression(8, 24)
# y = pw(y)
# assert not mridc.is_none(y)
# print(f"y: {y.shape}, x: {x.shape}")
# # cirim = mridc.nn.CIRIM(cfg=DictConfig({}))
# # assert not mridc.is_none(cirim)
# masker = mridc.create_masker("random1d", 0.04, 8)
# masked_kspace, mask, acc = mridc.apply_mask(y, masker)
# assert not mridc.is_none(masked_kspace)
# assert not mridc.is_none(mask)
# print(f"Acceleration: {acc}, mask: {mask.shape}, masked_kspace: {masked_kspace.shape}")
#
#
# plt.subplot(1, 3, 1)
# plt.imshow(torch.abs(torch.view_as_complex(x[0])), cmap="gray")
# plt.subplot(1, 3, 2)
# plt.imshow(torch.abs(torch.view_as_complex(y[0])), cmap="gray")
# plt.subplot(1, 3, 3)
# plt.imshow(torch.abs(torch.view_as_complex(masked_kspace[0])), cmap="gray")
# plt.show()
