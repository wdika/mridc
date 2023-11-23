# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

plt.style.use("dark_background")

wdir = "/data/projects/recon/other/dkarkalousos/UnSelfSupervised/plots/"
# files = list(Path("/scratch/iskylitsis/data/mridata_knee_2019/test/").iterdir())
files = list(
    Path(
        "/data/projects/recon/other/dkarkalousos/UnSelfSupervised/reconstructions_stanford_knees_poisson2d_12x/targets_SENSE/default/2023-04-18_17-21-19/reconstructions/"
    ).iterdir()
)
zf_path = "/data/projects/recon/other/dkarkalousos/UnSelfSupervised/reconstructions_stanford_knees_poisson2d_12x/ZF/default/2023-04-13_12-18-03/reconstructions/"
recons_paths = [
    "/data/projects/recon/other/dkarkalousos/UnSelfSupervised/reconstructions_stanford_knees_poisson2d_12x/CS/default/2023-04-13_15-09-35/reconstructions/",
    "/data/projects/recon/other/dkarkalousos/UnSelfSupervised/reconstructions_stanford_knees_poisson2d_12x/ProximalGradient_10ITER/default/2023-04-13_15-21-29/reconstructions/",
    "/data/projects/recon/other/dkarkalousos/UnSelfSupervised/reconstructions_stanford_knees_poisson2d_12x/N2R_1SUPSUBJ_stanford_knees_Poisson2d_12x/default/2023-04-17_10-02-02/reconstructions/",
    "/data/projects/recon/other/dkarkalousos/UnSelfSupervised/reconstructions_stanford_knees_poisson2d_12x/N2R_3SUPSUBJ_stanford_knees_Poisson2d_12x/default/2023-04-17_10-10-34/reconstructions/",
    "/data/projects/recon/other/dkarkalousos/UnSelfSupervised/reconstructions_stanford_knees_poisson2d_12x/N2R_FULLUNSUPSUP_stanford_knees_Poisson2d_12x/default/2023-04-17_10-23-35/reconstructions/",
    "/data/projects/recon/other/dkarkalousos/UnSelfSupervised/reconstructions_stanford_knees_poisson2d_12x/SSDU_RESNET_5UI1RB10CGITER_stanford_knees_Poisson2d_12x/default/2023-04-18_10-45-39/reconstructions/",
    "/data/projects/recon/other/dkarkalousos/UnSelfSupervised/reconstructions_stanford_knees_poisson2d_12x/SSDU_RESNET_5UI5RB10CGITER_stanford_knees_Poisson2d_12x/default/2023-04-17_10-14-00/reconstructions/",
    "/data/projects/recon/other/dkarkalousos/UnSelfSupervised/reconstructions_stanford_knees_poisson2d_12x/UNET_stanford_knees_Poisson2d_12x/2023-04-15_11-21-49/reconstructions/",
    "/data/projects/recon/other/dkarkalousos/UnSelfSupervised/reconstructions_stanford_knees_poisson2d_12x/UNET_3T_T1_3D_Brains_Gaussian2D_12x/2023-04-14_11-38-16/reconstructions/",
    "/data/projects/recon/other/dkarkalousos/UnSelfSupervised/reconstructions_stanford_knees_poisson2d_12x/RESNET_5UI5RB10CGITER_3T_T1_3D_Brains_Gaussian2D_12x/default/2023-04-15_11-31-12/reconstructions/",
    "/data/projects/recon/other/dkarkalousos/UnSelfSupervised/reconstructions_stanford_knees_poisson2d_12x/RESNET_5UI5RB10CGITER_stanford_knees_Poisson2d_12x/default/2023-04-18_11-15-23/reconstructions/",
    # "/data/projects/recon/other/dkarkalousos/UnSelfSupervised/reconstructions_stanford_knees_poisson2d_12x/IRIM_3T_T1_3D_Brains_Gaussian2D_12x/2023-04-15_11-07-39/reconstructions/",
    "/data/projects/recon/other/dkarkalousos/UnSelfSupervised/reconstructions_stanford_knees_poisson2d_12x/CIRIM_128F5C_3T_T1_3D_Brains_Gaussian2D_12x/2023-04-17_09-54-42/reconstructions/",
    # "/data/projects/recon/other/dkarkalousos/UnSelfSupervised/reconstructions_stanford_knees_poisson2d_12x/UNET_fastMRI_Knees_Gaussian2D_12x/2023-04-15_11-26-23/reconstructions/",
    # "/data/projects/recon/other/dkarkalousos/UnSelfSupervised/reconstructions_stanford_knees_poisson2d_12x/IRIM_fastMRI_Knees_Gaussian2D_12x/2023-04-15_11-15-02/reconstructions/",
]
slice_choices = [80, 160]

for file in tqdm(files):
    # _target = h5py.File(file, "r")["target"][()].squeeze()
    _target = h5py.File(file, "r")["reconstruction"][()].squeeze()
    _target = np.abs(_target / np.max(np.abs(_target)))
    for slice_choice in slice_choices:
        print(file.name, slice_choice)

        target = _target[slice_choice]
        # target = np.abs(target / np.max(np.abs(target)))
        target = [target, "Ground truth", "", ""]

        zero_filled = h5py.File(zf_path + "/" + file.name, "r")["reconstruction"][()].squeeze()
        zero_filled = np.abs(zero_filled / np.max(np.abs(zero_filled)))[slice_choice]
        # zero_filled = zero_filled[slice_choice]
        # zero_filled = np.abs(zero_filled / np.max(np.abs(zero_filled)))

        ssim = np.round(structural_similarity(zero_filled, target[0], data_range=1), 3)
        psnr = np.round(peak_signal_noise_ratio(zero_filled, target[0], data_range=1), 2)

        _zero_filled = [zero_filled, "Zero-Filled 12x", ssim, psnr]

        estimates = []
        for r in recons_paths:
            estimate = h5py.File(r + "/" + file.name, "r")["reconstruction"][()].squeeze()
            estimate = np.abs(estimate / np.max(np.abs(estimate)))[slice_choice]
            # estimate = estimate[slice_choice]
            # estimate = np.abs(estimate / np.max(np.abs(zero_filled)))

            name = ""
            if "CS" in r:
                name = "CS"
            elif "ProximalGradient" in r:
                name = "Proximal Gradient"
            elif "N2R_1SUPSUBJ" in r:
                name = "N2R (1SUBJ SS)"
            elif "N2R_3SUPSUBJ" in r:
                name = "N2R (3SUBJ SS)"
            elif "N2R_FULLUNSUPSUP" in r:
                name = "N2R (FULL SS)"
            elif "SSDU_RESNET_5UI5RB10CGITER" in r:
                name = "SSDU 5RB (SS)"
            elif "SSDU_RESNET_5UI1RB10CGITER" in r:
                name = "SSDU 1RB (SS)"
            elif "UNET_stanford_knees_Poisson2d_12x" in r:
                name = "UNET (S)"
            elif "UNET_3T_T1_3D_Brains_Gaussian2D_12x" in r:
                name = "UNET SPT"
            elif "RESNET_5UI5RB10CGITER_3T_T1_3D_Brains_Gaussian2D_12x" in r:
                name = "RESNET SPT"
            elif "RESNET_5UI5RB10CGITER_stanford_knees_Poisson2d_12x" in r:
                name = "RESNET (S)"
            elif "CIRIM_128F5C_3T_T1_3D_Brains_Gaussian2D_12x" in r:
                name = "CIRIM SPT"

            ssim = np.round(structural_similarity(estimate, target[0], data_range=1), 3)
            psnr = np.round(peak_signal_noise_ratio(estimate, target[0], data_range=1), 2)

            estimates.append([estimate, name, ssim, psnr])

        # iterate over estimates and sort them by SSIM
        _estimates = sorted(estimates, key=lambda x: x[2], reverse=True)

        # put ground truth first and "Zero-Filled 12x" second
        estimates = [target, _zero_filled] + _estimates

        # for slice_choice in range(target.shape[0]):
        fig, ax = plt.subplots(figsize=(12, 10))
        plt.axis("off")
        fig.patch.set_facecolor("black")
        fig.subplots_adjust(hspace=0.0, wspace=0.0)

        for i in range(len(estimates)):
            eta, name, ssim, psnr = estimates[i]

            if name not in ["Ground truth"]:
                ssim = "SSIM: " + str(ssim)
                psnr = "PSNR: " + str(psnr)

            ax = fig.add_subplot(3, 5, i + 1)
            ax.imshow(eta, cmap="gray")
            plt.text(0, 0, name, size=12, color="yellow")
            plt.text(0, 30, ssim, size=12, color="yellow")
            plt.text(0, 60, psnr, size=12, color="yellow")
            plt.axis("off")
            # set figure title

        # out_path = wdir + "/" + str(file.name)
        # Path(out_path).mkdir(exist_ok=True, parents=True)

        # plt.savefig(out_path + "/" + str(slice_choice) + ".png", dpi=300)
        plt.tight_layout()
        # plt.close()
        plt.show()
