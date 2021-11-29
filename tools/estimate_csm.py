# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

import argparse
import logging
import multiprocessing
import pathlib
import time
import numpy as np
from collections import defaultdict

from typing import Any, Optional, Tuple

import h5py

from mridc.data.mri_data import SliceDataset
from mridc.data.transforms import tensor_to_complex_np, to_tensor

import bart


class DataTransform:
    """Data transform for BART."""

    def __init__(self, split):
        self.retrieve_acc = split not in ("train", "val")

    def __call__(
        self, kspace, sensitivity_map, mask, eta, target, attrs, fname, slice_num
    ) -> Tuple[np.ndarray, str, int, Optional[Any]]:
        """

        Args:
            kspace: The k-space data.
            sensitivity_map: The sensitivity map.
            mask: The mask.
            eta: The initial estimation.
            target: The target.
            attrs: The attributes.
            fname: The filename.
            slice_num: The slice number.

        Returns:
            kspace: The k-space data.
            fname: The filename.
            slice_num: The slice number.
            num_low_freqs: The number of low frequencies.
        """
        num_low_freqs = attrs["num_low_frequency"] if self.retrieve_acc else None
        return (
            tensor_to_complex_np(to_tensor(kspace).permute(1, 2, 0, 3).unsqueeze(0).detach().cpu()),
            fname,
            slice_num,
            num_low_freqs,
        )


def ecalib(kspace: np.ndarray, num_low_freqs: int = None) -> np.ndarray:
    """
    Run ESPIRIT coil sensitivity estimation.
    Args:
        kspace: k-space data.
        num_low_freqs: Number of low frequencies.

    Returns:
        np.array: sense map.
    """
    if num_low_freqs is None:
        # if the following error occurs then the kernel size needs to be set manually
        # fftw: ../../kernel/planner.c:890: assertion failed: flags.l == l
        # sens_maps = bart.bart(1, "ecalib -d0 -m1 -k 10", kspace)
        sens_maps = bart.bart(1, "ecalib -d0 -m1", kspace)
    else:
        sens_maps = bart.bart(1, f"ecalib -d0 -m1 -r {num_low_freqs}", kspace)

    sens_maps = np.transpose(sens_maps[0], (2, 0, 1))

    return sens_maps


def save_outputs(outputs, output_path):
    """Saves reconstruction outputs to output_path."""
    reconstructions = defaultdict(list)

    for fname, slice_num, pred in outputs:
        reconstructions[fname].append((slice_num, pred))

    reconstructions = {
        fname: np.stack([pred for _, pred in sorted(slice_preds)]) for fname, slice_preds in reconstructions.items()
    }

    output_path.mkdir(exist_ok=True, parents=True)
    for fname, recons in reconstructions.items():
        with h5py.File(output_path / fname, "w") as hf:
            hf.create_dataset("sensitivity_map", data=recons)


def run_model(idx):
    """
    Run BART ecalib on idx index from dataset.

    Args:
        idx (int): The index of the dataset.

    Returns:
        tuple: tuple with
            fname: Filename
            slice_num: Slice number.
            sense_map: sense map.
    """
    masked_kspace, fname, slice_num, num_low_freqs = dataset[idx]
    sense_map = ecalib(masked_kspace, num_low_freqs)
    return fname, slice_num, sense_map


def run_bart(args):
    """Run the BART ecalib on the given data set."""
    if args.num_procs == 0:
        start_time = time.perf_counter()
        outputs = []
        for i in range(len(dataset)):
            outputs.append(run_model(i))
        time_taken = time.perf_counter() - start_time
    else:
        with multiprocessing.Pool(args.num_procs) as pool:
            start_time = time.perf_counter()
            outputs = pool.map(run_model, range(len(dataset)))
            time_taken = time.perf_counter() - start_time

    logging.info(f"Run Time = {time_taken:} s")
    save_outputs(outputs, args.output_path)


def create_arg_parser():
    """
    Create an argument parser.

    Returns:
        argparse.ArgumentParser: The argument parser.
    """
    parser = argparse.ArgumentParser(description="ECALIB")

    parser.add_argument("data_path", type=pathlib.Path, help="Path to the data folder")
    parser.add_argument("out_dir", type=pathlib.Path, help="Path to the output folder")
    parser.add_argument("--split", choices=["train", "val", "test", "challenge"], default="val", type=str)
    parser.add_argument(
        "--num_procs", type=int, default=4, help="Number of processes. Set to 0 to disable multiprocessing."
    )

    return parser


if __name__ == "__main__":
    ARGS = create_arg_parser().parse_args()

    # need this global for multiprocessing
    dataset = SliceDataset(
        root=ARGS.data_path,
        sense_root=None,
        transform=DataTransform(split=ARGS.split),
        challenge="multicoil",
        sample_rate=1.0,
    )

    run_bart(ARGS)
