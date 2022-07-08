# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

import argparse
import h5py
import numpy as np
import sys
import time
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt


def readcfl(name):
    with open(f"{name}.hdr", "r") as h:
        h.readline()  # skip
        l = h.readline()
    dims = [int(i) for i in l.split()]

    # remove singleton dimensions from the end
    n = int(np.prod(dims))
    dims_prod = np.cumprod(dims)
    dims = dims[: np.searchsorted(dims_prod, n) + 1]

    with open(f"{name}.cfl", "r") as d:
        a = np.fromfile(d, dtype=np.complex64, count=n)
    a = a.reshape(dims, order="F")  # column-major
    return a


def parse_file_path(file_path, str_match):
    """
    Parameters
    ----------
    file_path : Root file path to parse
    str_match : Name of the cfl file to check if kspace or coils sencitivities maps
    Returns
    -------
    A list with the filenames to convert
    """
    files = list(Path(file_path).iterdir())
    fnames = [str(files[i]).split("/")[-1] for i in range(len(files))]

    return sorted(
        [
            file_path + str(fnames[i]).split(".")[-2]
            for i in range(len(fnames))
            if str(fnames[i]).split("_")[-1] == f"{str_match}.cfl"
        ]
    )


def main(args):
    init_start = time.perf_counter()

    data = parse_file_path(file_path=args.file_path, str_match="kspace")
    csm = parse_file_path(file_path=args.file_path, str_match="sense")

    for i in tqdm(range(len(data))):
        print("Saving data to h5 format...\n")

        start_time = time.perf_counter()

        kspace = readcfl(data[i])
        imspace = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(kspace, axes=(0, 1, 2)), axes=(0, 1, 2)), axes=0)

        csm = readcfl(csm[i])
        csm = csm / np.sum(np.abs(csm), -1, keepdims=True)
        csm = csm / np.expand_dims(np.sqrt(np.sum(csm.conj() * csm, -1)), -1)
        csm[np.isnan(csm)] = 0 + 0j

        target = np.sum(imspace * csm.conj(), -1)
        norm = np.amax(np.abs(target).real.flatten())

        imspace = imspace / norm
        imspace = imspace / np.max(np.abs(imspace))
        imspace[np.isnan(imspace)] = 0 + 0j

        target = np.sum(imspace * csm.conj(), -1)

        kspace = np.fft.fftn(np.transpose(imspace, (2, 3, 0, 1)), axes=(-2, -1))
        csm = np.transpose(csm, (2, 3, 0, 1))
        target = np.transpose(target, (2, 0, 1))

        out_dir = Path("/".join(str(args.file_path).split("/")[:-2]) + "/h5/")
        out_dir.mkdir(parents=True, exist_ok=True)

        hf = h5py.File(Path(f"{str(out_dir)}/t1"), "w")
        hf.create_dataset("kspace", data=kspace.astype(np.complex64))
        hf.create_dataset("sensitivity_map", data=csm.astype(np.complex64))
        hf.create_dataset("target", data=np.abs(target).astype(np.float32))
        hf.close()

        print(
            "Done! Data saved into h5 format. It took",
            time.perf_counter() - start_time,
            "s \n",
        )


# noinspection PyTypeChecker
def create_arg_parser():
    parser = argparse.ArgumentParser(
        description="Convert cfl to numpy and then to selected format. By default data will be converted to h5.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("file_path", type=str, help="Path of the cfl file to be converted.")

    return parser


if __name__ == "__main__":
    args = create_arg_parser().parse_args(sys.argv[1:])
    args.out_dir = "/".join(args.file_path.split("/")[:-2])
    main(args)
