from pathlib import Path

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

out_dir = Path("/data/projects/recon/other/dkarkalousos/STAIRS/paper_figures/")
out_dir.mkdir(parents=True, exist_ok=True)

fontsize = 30

# Figure 1
img = np.asarray(Image.open("/data/projects/recon/other/dkarkalousos/STAIRS/png/subj01/axial/100.png"))
width = img.shape[1] // 3
zf = img[:, :width]
pics = img[:, width : 2 * width]
cirim = img[:, 2 * width :]
crop_height_start = 90
crop_height_end = 390
crop_width_start = 15
crop_width_end = 270
zf = zf[crop_height_start:crop_height_end, crop_width_start:crop_width_end]
pics = pics[crop_height_start:crop_height_end, crop_width_start:crop_width_end]
cirim = cirim[crop_height_start:crop_height_end, crop_width_start:crop_width_end]
zf = zf / np.max(zf)
zf = zf / np.median(zf)
zf = zf / np.max(zf)
zf = np.clip(zf, 0.1, 1)
zf = zf / np.max(zf)
pics = pics / np.max(pics)
pics = pics / np.median(pics)
pics = pics / np.max(pics)
pics = np.clip(pics, 0.0, 0.4)
pics = pics / np.max(pics)
cirim = cirim / np.max(cirim)
cirim = cirim / np.median(cirim)
cirim = cirim / np.max(cirim)
cirim = np.clip(cirim, 0.02, 0.4)
cirim = cirim / np.max(cirim)
fig1_img1 = np.concatenate([zf, pics, cirim], axis=1)

zf_label_x = 0
zf_label_y = 15
pics_label_x = abs(crop_width_start - crop_width_end)
pics_label_y = 15
cirim_label_x = abs(crop_width_start - crop_width_end) * 2
cirim_label_y = 15
fig = plt.figure(figsize=(20, 10))
fig.patch.set_facecolor("black")
plt.imshow(fig1_img1, cmap="gray")
plt.axis("off")
plt.text(zf_label_x, zf_label_y, "12x", color="yellow", fontsize=fontsize)
plt.text(pics_label_x, pics_label_y, "PICS", color="yellow", fontsize=fontsize)
plt.text(cirim_label_x, cirim_label_y, "CIRIM", color="yellow", fontsize=fontsize)
plt.savefig(str(out_dir) + "/subj01_axial_100.png", dpi=300)
# plt.show()

img = np.asarray(Image.open("/data/projects/recon/other/dkarkalousos/STAIRS/png/subj09/sagittal/80.png"))
width = img.shape[1] // 3
zf = img[:, :width]
pics = img[:, width : 2 * width]
cirim = img[:, 2 * width :]
crop_height_start = 50
crop_height_end = 310
crop_width_start = 170
crop_width_end = 490
zf = zf[crop_height_start:crop_height_end, crop_width_start:crop_width_end]
pics = pics[crop_height_start:crop_height_end, crop_width_start:crop_width_end]
cirim = cirim[crop_height_start:crop_height_end, crop_width_start:crop_width_end]
zf = zf / np.max(zf)
zf = zf / np.median(zf)
zf = zf / np.max(zf)
zf = zf / np.max(zf)
pics = pics / np.max(pics)
pics = pics / np.median(pics)
pics = pics / np.max(pics)
pics = pics / np.max(pics)
cirim = cirim / np.max(cirim)
cirim = cirim / np.median(cirim)
cirim = cirim / np.max(cirim)
cirim = np.clip(cirim, 0.05, 1)
cirim = cirim / np.max(cirim)
fig1_img2 = np.concatenate([zf, pics, cirim], axis=1)

zf_label_x = 0
zf_label_y = 15
pics_label_x = abs(crop_width_start - crop_width_end)
pics_label_y = 15
cirim_label_x = abs(crop_width_start - crop_width_end) * 2
cirim_label_y = 15
fig = plt.figure(figsize=(20, 10))
fig.patch.set_facecolor("black")
plt.imshow(fig1_img2, cmap="gray")
plt.axis("off")
plt.text(zf_label_x, zf_label_y, "12x", color="yellow", fontsize=fontsize)
plt.text(pics_label_x, pics_label_y, "PICS", color="yellow", fontsize=fontsize)
plt.text(cirim_label_x, cirim_label_y, "CIRIM", color="yellow", fontsize=fontsize)
plt.savefig(str(out_dir) + "/subj09_sagittal_80.png", dpi=300)

img = np.asarray(Image.open("/data/projects/recon/other/dkarkalousos/STAIRS/png/subj09/coronal/160.png"))
width = img.shape[1] // 3
zf = img[:, :width]
pics = img[:, width : 2 * width]
cirim = img[:, 2 * width :]
crop_height_start = 15
crop_height_end = 345
crop_width_start = 10
crop_width_end = 345
zf = zf[crop_height_start:crop_height_end, crop_width_start:crop_width_end]
pics = pics[crop_height_start:crop_height_end, crop_width_start:crop_width_end]
cirim = cirim[crop_height_start:crop_height_end, crop_width_start:crop_width_end]
zf = zf / np.max(zf)
zf = zf / np.median(zf)
zf = zf / np.max(zf)
zf = zf / np.max(zf)
pics = pics / np.max(pics)
pics = pics / np.median(pics)
pics = pics / np.max(pics)
pics = pics / np.max(pics)
cirim = cirim / np.max(cirim)
cirim = cirim / np.median(cirim)
cirim = cirim / np.max(cirim)
cirim = cirim / np.max(cirim)
fig1_img3 = np.concatenate([zf, pics, cirim], axis=1)

zf_label_x = 0
zf_label_y = 15
pics_label_x = abs(crop_width_start - crop_width_end)
pics_label_y = 15
cirim_label_x = abs(crop_width_start - crop_width_end) * 2
cirim_label_y = 15
fig = plt.figure(figsize=(20, 10))
fig.patch.set_facecolor("black")
plt.imshow(fig1_img3, cmap="gray")
plt.axis("off")
plt.text(zf_label_x, zf_label_y, "12x", color="yellow", fontsize=fontsize)
plt.text(pics_label_x, pics_label_y, "PICS", color="yellow", fontsize=fontsize)
plt.text(cirim_label_x, cirim_label_y, "CIRIM", color="yellow", fontsize=fontsize)
plt.savefig(str(out_dir) + "/subj09_coronal_160.png", dpi=300)

# Figure 2
img = np.asarray(Image.open("/data/projects/recon/other/dkarkalousos/STAIRS/png/subj10/coronal/175.png"))
width = img.shape[1] // 3
zf = img[:, :width]
pics = img[:, width : 2 * width]
cirim = img[:, 2 * width :]
crop_height_start = 20
crop_height_end = 350
crop_width_start = 20
crop_width_end = 340
zf = zf[crop_height_start:crop_height_end, crop_width_start:crop_width_end]
pics = pics[crop_height_start:crop_height_end, crop_width_start:crop_width_end]
cirim = cirim[crop_height_start:crop_height_end, crop_width_start:crop_width_end]
zf = zf / np.max(zf)
zf = zf / np.median(zf)
zf = zf / np.max(zf)
zf = np.clip(zf, 0.15, 1)
zf = zf / np.max(zf)
pics = pics / np.max(pics)
pics = pics / np.median(pics)
pics = pics / np.max(pics)
pics = np.clip(pics, 0.05, 0.8)
pics = pics / np.max(pics)
cirim = cirim / np.max(cirim)
cirim = cirim / np.median(cirim)
cirim = cirim / np.max(cirim)
cirim = np.clip(cirim, 0.12, 0.88)
cirim = cirim / np.max(cirim)
fig2_img1 = np.concatenate([zf, pics, cirim], axis=1)

zf_label_x = 0
zf_label_y = 15
pics_label_x = abs(crop_width_start - crop_width_end)
pics_label_y = 15
cirim_label_x = abs(crop_width_start - crop_width_end) * 2
cirim_label_y = 15
fig = plt.figure(figsize=(20, 10))
fig.patch.set_facecolor("black")
plt.imshow(fig2_img1, cmap="gray")
plt.axis("off")
plt.text(zf_label_x, zf_label_y, "12x", color="yellow", fontsize=fontsize)
plt.text(pics_label_x, pics_label_y, "PICS", color="yellow", fontsize=fontsize)
plt.text(cirim_label_x, cirim_label_y, "CIRIM", color="yellow", fontsize=fontsize)
plt.savefig(str(out_dir) + "/subj10_coronal_slice_175.png", dpi=300)

img = np.asarray(Image.open("/data/projects/recon/other/dkarkalousos/STAIRS/png/subj16/sagittal/225.png"))
width = img.shape[1] // 3
zf = img[:, :width]
pics = img[:, width : 2 * width]
cirim = img[:, 2 * width :]
crop_height_start = 20
crop_height_end = 330
crop_width_start = 70
crop_width_end = 480
zf = zf[crop_height_start:crop_height_end, crop_width_start:crop_width_end]
pics = pics[crop_height_start:crop_height_end, crop_width_start:crop_width_end]
cirim = cirim[crop_height_start:crop_height_end, crop_width_start:crop_width_end]
zf = zf / np.max(zf)
zf = zf / np.median(zf)
zf = zf / np.max(zf)
zf = np.clip(zf, 0.1, 1)
zf = zf / np.max(zf)
pics = pics / np.max(pics)
pics = pics / np.median(pics)
pics = pics / np.max(pics)
pics = pics / np.max(pics)
cirim = cirim / np.max(cirim)
cirim = cirim / np.median(cirim)
cirim = cirim / np.max(cirim)
cirim = cirim / np.max(cirim)
fig2_img2 = np.concatenate([zf, pics, cirim], axis=1)

zf_label_x = 0
zf_label_y = 20
pics_label_x = abs(crop_width_start - crop_width_end)
pics_label_y = 20
cirim_label_x = abs(crop_width_start - crop_width_end) * 2
cirim_label_y = 20
fig = plt.figure(figsize=(20, 10))
fig.patch.set_facecolor("black")
plt.imshow(fig2_img2, cmap="gray")
plt.axis("off")
plt.text(zf_label_x, zf_label_y, "12x", color="yellow", fontsize=fontsize)
plt.text(pics_label_x, pics_label_y, "PICS", color="yellow", fontsize=fontsize)
plt.text(cirim_label_x, cirim_label_y, "CIRIM", color="yellow", fontsize=fontsize)
plt.savefig(str(out_dir) + "/subj16_sagittal_slice_225.png", dpi=300)

img = np.asarray(Image.open("/data/projects/recon/other/dkarkalousos/STAIRS/png/subj06/axial/120.png"))
width = img.shape[1] // 3
zf = img[:, :width]
pics = img[:, width : 2 * width]
cirim = img[:, 2 * width :]
crop_height_start = 35
crop_height_end = 315
crop_width_start = 50
crop_width_end = 300
zf = zf[crop_height_start:crop_height_end, crop_width_start:crop_width_end]
pics = pics[crop_height_start:crop_height_end, crop_width_start:crop_width_end]
cirim = cirim[crop_height_start:crop_height_end, crop_width_start:crop_width_end]
zf = zf / np.max(zf)
zf = zf / np.median(zf)
zf = zf / np.max(zf)
zf = zf / np.max(zf)
pics = pics / np.max(pics)
pics = pics / np.median(pics)
pics = pics / np.max(pics)
pics = pics / np.max(pics)
cirim = cirim / np.max(cirim)
cirim = cirim / np.median(cirim)
cirim = cirim / np.max(cirim)
cirim = cirim / np.max(cirim)
fig2_img3 = np.concatenate([zf, pics, cirim], axis=1)

zf_label_x = 0
zf_label_y = 15
pics_label_x = abs(crop_width_start - crop_width_end)
pics_label_y = 15
cirim_label_x = abs(crop_width_start - crop_width_end) * 2
cirim_label_y = 15
fig = plt.figure(figsize=(20, 10))
fig.patch.set_facecolor("black")
plt.imshow(fig2_img3, cmap="gray")
plt.axis("off")
plt.text(zf_label_x, zf_label_y, "12x", color="yellow", fontsize=fontsize)
plt.text(pics_label_x, pics_label_y, "PICS", color="yellow", fontsize=fontsize)
plt.text(cirim_label_x, cirim_label_y, "CIRIM", color="yellow", fontsize=fontsize)
plt.savefig(str(out_dir) + "/subj06_axial_120.png", dpi=300)

# Figure 3
img = np.asarray(Image.open("/data/projects/recon/other/dkarkalousos/STAIRS/png/subj05/axial/150.png"))
width = img.shape[1] // 3
zf = img[:, :width]
pics = img[:, width : 2 * width]
cirim = img[:, 2 * width :]
crop_height_start = 70
crop_height_end = 390
crop_width_start = 15
crop_width_end = 270
zf = zf[crop_height_start:crop_height_end, crop_width_start:crop_width_end]
pics = pics[crop_height_start:crop_height_end, crop_width_start:crop_width_end]
cirim = cirim[crop_height_start:crop_height_end, crop_width_start:crop_width_end]
zf = zf / np.max(zf)
zf = zf / np.median(zf)
zf = zf / np.max(zf)
zf = zf / np.max(zf)
pics = pics / np.max(pics)
pics = pics / np.median(pics)
pics = pics / np.max(pics)
pics = pics / np.max(pics)
cirim = cirim / np.max(cirim)
cirim = cirim / np.median(cirim)
cirim = cirim / np.max(cirim)
cirim = cirim / np.max(cirim)
fig3_img1 = np.concatenate([zf, pics, cirim], axis=1)

zf_label_x = 0
zf_label_y = 15
pics_label_x = abs(crop_width_start - crop_width_end)
pics_label_y = 15
cirim_label_x = abs(crop_width_start - crop_width_end) * 2
cirim_label_y = 15
fig = plt.figure(figsize=(20, 10))
fig.patch.set_facecolor("black")
plt.imshow(fig3_img1, cmap="gray")
plt.axis("off")
plt.text(zf_label_x, zf_label_y, "12x", color="yellow", fontsize=fontsize)
plt.text(pics_label_x, pics_label_y, "PICS", color="yellow", fontsize=fontsize)
plt.text(cirim_label_x, cirim_label_y, "CIRIM", color="yellow", fontsize=fontsize)
plt.savefig(str(out_dir) + "/subj05_axial_150.png", dpi=300)

img = np.asarray(Image.open("/data/projects/recon/other/dkarkalousos/STAIRS/png/subj04/sagittal/102.png"))
width = img.shape[1] // 3
zf = img[:, :width]
pics = img[:, width : 2 * width]
cirim = img[:, 2 * width :]
crop_height_start = 5
crop_height_end = 250
crop_width_start = 90
crop_width_end = 380
zf = zf[crop_height_start:crop_height_end, crop_width_start:crop_width_end]
pics = pics[crop_height_start:crop_height_end, crop_width_start:crop_width_end]
cirim = cirim[crop_height_start:crop_height_end, crop_width_start:crop_width_end]
zf = zf / np.max(zf)
zf = zf / np.median(zf)
zf = zf / np.max(zf)
zf = np.clip(zf, 0.1, 1)
zf = zf / np.max(zf)
pics = pics / np.max(pics)
pics = pics / np.median(pics)
pics = pics / np.max(pics)
pics = pics / np.max(pics)
cirim = cirim / np.max(cirim)
cirim = cirim / np.median(cirim)
cirim = cirim / np.max(cirim)
cirim = np.clip(cirim, 0.0, 0.9)
cirim = cirim / np.median(cirim)
cirim = cirim / np.max(cirim)
fig3_img2 = np.concatenate([zf, pics, cirim], axis=1)

zf_label_x = 0
zf_label_y = 20
pics_label_x = abs(crop_width_start - crop_width_end)
pics_label_y = 20
cirim_label_x = abs(crop_width_start - crop_width_end) * 2
cirim_label_y = 20
fig = plt.figure(figsize=(20, 10))
fig.patch.set_facecolor("black")
plt.imshow(fig3_img2, cmap="gray")
plt.axis("off")
plt.text(zf_label_x, zf_label_y, "12x", color="yellow", fontsize=fontsize)
plt.text(pics_label_x, pics_label_y, "PICS", color="yellow", fontsize=fontsize)
plt.text(cirim_label_x, cirim_label_y, "CIRIM", color="yellow", fontsize=fontsize)
plt.savefig(str(out_dir) + "/subj04_sagittal_102.png", dpi=300)

img = np.asarray(Image.open("/data/projects/recon/other/dkarkalousos/STAIRS/png/subj01/coronal/248.png"))
width = img.shape[1] // 3
zf = img[:, :width]
pics = img[:, width : 2 * width]
cirim = img[:, 2 * width :]
crop_height_start = 30
crop_height_end = 300
crop_width_start = 10
crop_width_end = 340
zf = zf[crop_height_start:crop_height_end, crop_width_start:crop_width_end]
pics = pics[crop_height_start:crop_height_end, crop_width_start:crop_width_end]
cirim = cirim[crop_height_start:crop_height_end, crop_width_start:crop_width_end]
zf = zf / np.max(zf)
zf = zf / np.median(zf)
zf = zf / np.max(zf)
zf = zf / np.max(zf)
pics = pics / np.max(pics)
pics = pics / np.median(pics)
pics = pics / np.max(pics)
pics = pics / np.max(pics)
cirim = cirim / np.max(cirim)
cirim = cirim / np.median(cirim)
cirim = cirim / np.max(cirim)
cirim = cirim / np.max(cirim)
fig3_img3 = np.concatenate([zf, pics, cirim], axis=1)

zf_label_x = 0
zf_label_y = 20
pics_label_x = abs(crop_width_start - crop_width_end) - 60
pics_label_y = 20
cirim_label_x = abs(crop_width_start - crop_width_end) * 2 - 120
cirim_label_y = 20
fig = plt.figure(figsize=(20, 10))
fig.patch.set_facecolor("black")
plt.imshow(fig3_img3, cmap="gray")
plt.axis("off")
plt.text(zf_label_x, zf_label_y, "12x", color="yellow", fontsize=fontsize)
plt.text(pics_label_x, pics_label_y, "PICS", color="yellow", fontsize=fontsize)
plt.text(cirim_label_x, cirim_label_y, "CIRIM", color="yellow", fontsize=fontsize)
plt.savefig(str(out_dir) + "/subj01_coronal_248.png", dpi=300)
