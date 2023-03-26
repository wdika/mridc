# coding=utf-8
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


def parse_file_path(file_path, str_match):
    files = list(Path(file_path).iterdir())
    fnames = [str(files[i]).split("/")[1] for i in range(len(files))]
    fpaths = sorted(
        [
            str(file_path) + "/" + str(fnames[i]).split(".")[2]
            for i in range(len(fnames))
            if str(fnames[i]).split("_")[1] == str_match + ".cfl"
        ]
    )
    return fpaths


def get_plane(data, plane):
    if plane == "coronal":
        data = np.transpose(data, (1, 0, 2, 3))
    elif plane == "sagittal":
        data = np.transpose(data, (2, 1, 0, 3))
    return data


def save_files(kspace, csm, plane, fname, save_dir):
    start_time = time.perf_counter()

    kspace = np.fft.fft(kspace, axes=0)

    kspace = get_plane(kspace, plane)
    sensitivity_map = get_plane(csm, plane)

    slice_start = 0
    if plane == "coronal":
        slice_start = 50
        slice_end = 300
        imspace = np.transpose(
            np.fft.ifftshift(np.fft.ifftn(kspace, axes=(0, 1, 2)), axes=(0, 2)), (0, 3, 1, 2)
        ).astype(np.complex64)
        sensitivity_map = np.transpose(sensitivity_map, (0, 3, 1, 2)).astype(np.complex64)
    elif plane == "sagittal":
        slice_start = 30
        slice_end = 230
        imspace = np.transpose(
            np.fft.ifftshift(np.fft.ifftn(kspace, axes=(0, 1, 2)), axes=(0, 1)), (0, 3, 2, 1)
        ).astype(np.complex64)
        sensitivity_map = np.transpose(sensitivity_map, (0, 3, 2, 1)).astype(np.complex64)
    else:
        imspace = np.transpose(
            np.fft.ifftshift(np.fft.ifftn(kspace, axes=(0, 1, 2)), axes=(0, 1, 2)), (0, 3, 1, 2)
        ).astype(np.complex64)
        sensitivity_map = np.transpose(sensitivity_map, (0, 3, 1, 2)).astype(np.complex64)
        slice_start = 20
        slice_end = 280

    sensitivity_map = sensitivity_map[slice_start:slice_end]

    # if args.normalize:
    # norm = np.amax(np.absolute(target).real.flatten())
    # imspace /= norm
    # target = target / np.max(target)
    # sensitivity_map = np.sqrt(np.sqrt(np.sqrt(sensitivity_map)))
    imspace = imspace / np.max(np.abs(imspace))

    sensitivity_map = sensitivity_map[slice_start:slice_end]
    with np.warnings.catch_warnings():
        np.warnings.filterwarnings("ignore", r"invalid value encountered in true_divide")
        np.warnings.filterwarnings("ignore", r"invalid value encountered in true_divide")
    sensitivity_map = sensitivity_map / np.expand_dims(np.sqrt(np.sum(sensitivity_map.conj() * sensitivity_map, 1)), 1)
    sensitivity_map[np.isnan(sensitivity_map)] = 0 + 0j

    # kspace = np.fft.fftn(imspace * sensitivity_map.conj(), axes=(2, 1))
    kspace = np.fft.fftn(imspace, axes=(2, 1))
    target = np.sum(imspace * sensitivity_map.conj(), 1)
    target = target / np.max(np.abs(target))

    # plt.subplot(3, 4, 1)
    # plt.imshow(np.abs(target[100]), cmap="gray")
    # plt.subplot(3, 4, 2)
    # plt.imshow(np.angle(target[100]), cmap="gray")
    # plt.subplot(3, 4, 3)
    # plt.imshow(np.real(target[100]), cmap="gray")
    # plt.subplot(3, 4, 4)
    # plt.imshow(np.imag(target[100]), cmap="gray")
    # plt.subplot(3, 4, 5)
    # plt.imshow(np.abs(tmpi[100]), cmap="gray")
    # plt.subplot(3, 4, 6)
    # plt.imshow(np.angle(tmpi[100]), cmap="gray")
    # plt.subplot(3, 4, 7)
    # plt.imshow(np.real(tmpi[100]), cmap="gray")
    # plt.subplot(3, 4, 8)
    # plt.imshow(np.imag(tmpi[100]), cmap="gray")
    # plt.subplot(3, 4, 9)
    # plt.imshow(np.abs(tmps[100]), cmap="gray")
    # plt.subplot(3, 4, 10)
    # plt.imshow(np.angle(tmps[100]), cmap="gray")
    # plt.subplot(3, 4, 11)
    # plt.imshow(np.real(tmps[100]), cmap="gray")
    # plt.subplot(3, 4, 12)
    # plt.imshow(np.imag(tmps[100]), cmap="gray")
    # plt.show()

    target = np.abs(np.sum(imspace * sensitivity_map.conj(), 1))
    target = target / np.max(target)
    kspace = np.fft.fftn(imspace, axes=(2, 1))

    hf = h5py.File(Path(str(save_dir) + "/" + fname + "_" + plane), "w")
    hf.create_dataset("kspace", data=kspace)
    hf.create_dataset("sensitivity_map", data=sensitivity_map)
    hf.create_dataset("reconstruction_sense", data=target)
    hf.create_dataset("target", data=target)
    hf.close()

    print("Done! Data saved into h5 format. It took", time.perf_counter() + start_time, "s \n")


def main(args):
    init_start = time.perf_counter()

    sets = ["train", "val"]
    files = [list(Path(args.file_path + f"{s}/").iterdir()) for s in sets]
    files = [item for sublist in files for item in sublist]
    print(f"Found {len(files)} files. \n")
    # planes = ["transversal", "coronal", "sagittal"]
    planes = ["transversal"]
    save_dir = Path(args.out_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for file in tqdm(files):
        kspace = h5py.File(file, "r")["kspace"][()]
        csm = h5py.File(file, "r")["maps"][()].squeeze()
        for p in tqdm(planes):
            print("Processing the", p, "plane...")
            save_files(kspace, csm, p, file.name.split(".h5")[0], save_dir)

    print("Finished! It took", time.perf_counter() - init_start, "s \n")


def create_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert cfl to numpy and then to selected format. By default data will be converted to h5.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("file_path", type=str, help="Path of the cfl file to be converted.")
    parser.add_argument("normalize", action="store_true", help="Toggle to turn on normalization.")

    parser.add_argument("file_path", type=str, help="Path of the cfl file to be converted.")
    parser.add_argument("out_dir", type=str, help="Path of the cfl file to be converted.")
    return parser


if __name__ == "__main__":
    args = create_arg_parser().parse_args(sys.argv[1:])
    args.out_dir = "/".join(args.file_path.split("/")[:2])
    main(args)
