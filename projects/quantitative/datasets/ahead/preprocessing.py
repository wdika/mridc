# coding=utf-8
__author__ = "Chaoping Zhang, Dimitrios Karkalousos"

import argparse
import glob
import os
import sys
from pathlib import Path

import SimpleITK as sitk
import h5py
import numpy as np
import torch
from skimage.restoration import unwrap_phase
from tqdm import tqdm

from mridc.collections.quantitative.parts.transforms import LeastSquares


def _dataloder(subjectID: str, datapath: str):
    """
    Load coil images, sensitivity maps, and brain mask.

    Parameters
    ----------
    subjectID : str
        Subject ID.
    datapath : str
        Path to the data.

    Returns
    -------
    coilimgs : numpy array
        Coil images.
    sense : numpy array
        Sensitivity maps.
    mask_brain : numpy array
        Brain mask.
    """
    sense_complex = False
    coilimgs = False
    brain_mask = False
    if folders := glob.glob(f"{datapath}Subcortex_{subjectID.zfill(4)}*_R02"):
        filename_sense = glob.glob(
            os.path.join(
                folders[0],
                f"Subcortex_{subjectID.zfill(4)}*_R02_inv2_rcal.mat",
            )
        )

        file_coilimgs_p1 = f"Subcortex_{subjectID.zfill(4)}*_R02_inv2_"
        file_coilimgs_p2 = "_gdataCorrected.nii.gz"
        filename_coilimgs = glob.glob(os.path.join(folders[0], f"{file_coilimgs_p1}*{file_coilimgs_p2}"))

        if filename_sense and filename_coilimgs:
            # load sensitivity map (complex-valued)
            with h5py.File(filename_sense[0], "r") as f:
                for k, v in f.items():
                    sense = np.array(v)
                    sense_complex = sense["real"] + 1j * sense["imag"]
                    sense_complex = np.transpose(sense_complex, (3, 2, 1, 0))

            # load coil images (complex-valued)
            coilimgs = []  # type: ignore
            for i in range(1, 5):  # type: ignore
                filename_coilimg = glob.glob(os.path.join(folders[0], file_coilimgs_p1 + str(i) + file_coilimgs_p2))
                coilimg = sitk.ReadImage(filename_coilimg[0])
                coilimgs.append(np.transpose(sitk.GetArrayFromImage(coilimg), (3, 2, 1, 0)))  # type: ignore

            # load brain mask
            brain_mask = sitk.ReadImage(os.path.join(folders[0], "nii", "mask_inv2_te2_m_corr.nii"))
            brain_mask = sitk.GetArrayFromImage(brain_mask)
            brain_mask = np.flip(np.transpose(brain_mask, (0, 2, 1)), 1)  # need to flip! in the second axis!

    return coilimgs, sense_complex, brain_mask


def B0mapping(coilimgs, sense, mask_brain):
    """
    B0 mapping.

    Parameters
    ----------
    coilimgs : numpy array
        Coil images.
    sense : numpy array
        Sensitivity maps.
    mask_brain : numpy array
        Brain mask.

    Returns
    -------
    B0map : numpy array
        B0 map.
    """
    TEnotused = 3

    imgs = np.sum(coilimgs * sense.conj(), -1)

    phases = np.angle(imgs)
    phase_unwrapped = np.zeros(phases.shape)
    for i in range(phases.shape[0]):
        phase_unwrapped[i] = unwrap_phase(np.ma.array(phases[i], mask=np.zeros(phases[i].shape)))

    TEs = (3.0, 11.5, 20.0, 28.5)
    phase_diff_set = []
    TE_diff = []
    # obtain phase differences and TE differences
    for i in range(phase_unwrapped.shape[0] - TEnotused):
        phase_diff_set.append((phase_unwrapped[i + 1] - phase_unwrapped[i]).flatten())
        phase_diff_set[i] = (
            phase_diff_set[i]
            - np.round(np.sum(phase_diff_set[i] * mask_brain.flatten()) / np.sum(mask_brain.flatten()) / 2 / np.pi)
            * 2
            * np.pi
        )
        TE_diff.append(TEs[i + 1] - TEs[i])

    phase_diff_set = np.stack(phase_diff_set, 0)
    TE_diff = np.stack(TE_diff, 0)

    # least squares fitting to obtain phase map
    scaling = 1e-3
    ls = LeastSquares()
    B0_map_tmp = ls.lstq_pinv(
        torch.from_numpy(np.transpose(np.expand_dims(phase_diff_set, 2), (1, 0, 2))),
        torch.from_numpy(np.expand_dims(TE_diff, 1) * scaling),
    )
    B0_map = B0_map_tmp.reshape(phase_unwrapped.shape[1:4])
    B0_map = B0_map.numpy()

    return B0_map


def generate_2dksp(images3d, dim2keep):
    """
    Generate 2D k-space.

    Parameters
    ----------
    images3d : numpy array
        3D images.
    dim2keep : int
        Dimension to keep.

    Returns
    -------
    ksp2d : numpy array
        2D k-space.
    """
    axes = [[2, 3], [1, 3], [1, 2]]
    return np.fft.fftshift(np.fft.fft2(images3d, axes=axes[dim2keep], norm="ortho"), axes=axes[dim2keep])


def main(args):
    # loop over all subjects
    # process data for one subject:
    # 1. load one subject: coil images, sensitivity maps.
    # 2. create 3D B0 map: a. obtain 3D phase; b. unwrap phase; c. compute B0 map (use first 2 echoes).
    # 3. create 2D kspaces. a. sagittal; b. axial; c. coronal.
    # 4. load brain mask.
    # 5. save as .h5 for each slice in each plane of the 3 orientations.
    # loop over all orientations.
    # loop over all slices in one orientation.

    datapath = args.datapath
    applymask = args.applymask
    if not applymask:
        centerslices = args.centerslices
        half_nr_of_slices = 25 if centerslices else 50
    savepath = args.savepath
    for subjectID in tqdm(range(1, 119)):
        coilimgs, sense, brain_mask = _dataloder(subjectID, datapath)
        if coilimgs != False:
            coilimgs = np.stack(coilimgs, axis=0)
            if applymask:
                coilimgs = coilimgs * np.repeat(brain_mask[..., np.newaxis], coilimgs.shape[-1], axis=3)
                sense = sense * np.repeat(brain_mask[..., np.newaxis], sense.shape[-1], axis=3)
            B0map = B0mapping(coilimgs, sense, brain_mask)
            planes = ["sagittal", "coronal", "axial"]
            folder_subject = f"Subcortex_{str(subjectID).zfill(4)}_R02_inv2"
            for dim in range(3):
                ksp = generate_2dksp(coilimgs, dim)
                ksp_dim = np.swapaxes(ksp, 1, dim + 1)
                sense_dim = np.swapaxes(sense, 0, dim)
                B0map_dim = np.swapaxes(B0map, 0, dim)
                brain_mask_dim = np.swapaxes(brain_mask, 0, dim)
                size_dim = coilimgs.shape[dim + 1]
                Path(os.path.join(savepath, folder_subject, planes[dim])).mkdir(parents=True, exist_ok=True)
                for itr_dim in range(round(size_dim / 2) - half_nr_of_slices, round(size_dim / 2) + half_nr_of_slices):
                    filename_save = os.path.join(
                        savepath,
                        folder_subject,
                        planes[dim],
                        f"Subcortex_{str(subjectID).zfill(4)}_{planes[dim]}_{str(itr_dim)}.h5",
                    )

                    with h5py.File(filename_save, "w") as data:
                        data.create_dataset("ksp", data=ksp_dim[:, itr_dim, ...].squeeze())
                        data.create_dataset("sense", data=sense_dim[itr_dim, ...].squeeze())
                        data.create_dataset("B0map", data=B0map_dim[itr_dim, ...].squeeze())
                        data.create_dataset("mask_brain", data=brain_mask_dim[itr_dim, ...].squeeze())


# noinspection PyTypeChecker
def create_arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("datapath", type=str, help="Path of the files to be converted.")
    parser.add_argument("savepath", type=str, help="Path to save the converted files.")
    parser.add_argument("--applymask", action="store_true", help="Apply brain mask.")
    parser.add_argument("--centerslices", action="store_true", help="Save center slices.")
    return parser


if __name__ == "__main__":
    args = create_arg_parser().parse_args(sys.argv[1:])
    main(args)
