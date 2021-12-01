# coding=utf-8
__author__ = "Dimitrios Karkalousos"

import argparse
import gc
import pathlib
import sys
import time
from collections import defaultdict
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from mridc import save_reconstructions, complex_mul, complex_conj, ifft2c
from mridc.data.mri_data import SliceDataset
from mridc.data.subsample import create_mask_for_mask_type
from mridc.data.transforms import center_crop_to_smallest, PhysicsInformedDataTransform

torch.backends.cudnn.benchmark = False


def run_zf(data_loader: DataLoader, fft_type: str, device: str, progress_bar_refresh: int) -> Dict:
    """
    Run the zero-filled reconstructions.

    Args:
        data_loader: The data loader.
        fft_type: The FFT type.
        device: The device to use.
        progress_bar_refresh: The progress bar refresh rate.

    Returns:
        Dict with the reconstructions.
    """
    output = defaultdict(list)
    for i, data in enumerate(data_loader):
        (masked_kspace, sensitivity_map, _, target, fname, slice_num, _, _, _) = data
        masked_kspace = masked_kspace.to(device)
        sensitivity_map = sensitivity_map.to(device)
        target = target.to(device)

        # run the forward pass
        with torch.no_grad():
            estimate = torch.abs(
                torch.view_as_complex(
                    torch.sum(complex_mul(ifft2c(masked_kspace, fft_type=fft_type), complex_conj(sensitivity_map)), 1)
                )
            )
            target, estimate = center_crop_to_smallest(target, estimate)

            output[fname[0]].append((slice_num, estimate.cpu()))

        gc.collect()
        torch.cuda.empty_cache()

        # update the progress bar
        if i % progress_bar_refresh == 0:
            sys.stdout.write("\r[{:5.2f}%]".format(100 * (i + 1) / len(data_loader)))
            sys.stdout.flush()

    sys.stdout.write("\n")
    sys.stdout.flush()

    return {fname: np.stack([pred for _, pred in sorted(slice_preds)]) for fname, slice_preds in output.items()}


def main(args):
    """
    Main function.

    Args:
        args: The command line arguments.

    Returns:
        None.
    """
    data_loader = DataLoader(
        dataset=SliceDataset(
            root=args.data_path,
            sense_root=args.sense_path,
            challenge=args.challenge,
            transform=PhysicsInformedDataTransform(
                mask_func=False
                if args.no_mask
                else create_mask_for_mask_type(args.mask_type, args.center_fractions, args.accelerations),
                shift_mask=args.shift_mask,
                normalize_inputs=args.normalize_inputs,
                crop_size=args.crop_size,
                crop_before_masking=args.crop_before_masking,
                kspace_zero_filling_size=args.kspace_zero_filling_size,
                fft_type=args.fft_type,
            ),
            sample_rate=args.sample_rate,
            mask_root=args.mask_path,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    init_start = time.perf_counter()

    # TODO: change print to logger
    print("Reconstructing...")
    reconstructions = run_zf(data_loader, args.fft_type, args.device, args.progress_bar_refresh)

    print("Saving...")
    save_reconstructions(reconstructions, args.out_dir / "reconstructions")

    print("Finished! It took", time.perf_counter() - init_start, "s \n")


def create_arg_parser():
    """
    Creates an ArgumentParser to read the arguments.

    Returns:
        argparse.ArgumentParser: An ArgumentParser object.
    """
    parser = argparse.ArgumentParser(description="ZERO-FILLED")

    parser.add_argument("data_path", type=pathlib.Path, help="Path to the data folder")
    parser.add_argument("out_dir", type=pathlib.Path, help="Path to the output folder")
    parser.add_argument("--sense_path", type=pathlib.Path, help="Path to the sense folder")
    parser.add_argument("--mask_path", type=pathlib.Path, help="Path to the mask folder")
    parser.add_argument(
        "--data-split",
        choices=["val", "test", "test_v2", "challenge"],
        help='Which data partition to run on: "val" or "test"',
    )
    parser.add_argument(
        "--challenge",
        type=str,
        choices=["singlecoil", "multicoil"],
        default="multicoil",
        help="Which challenge to run",
    )
    parser.add_argument("--sample_rate", type=float, default=1.0, help="Sample rate for the data")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for the data loader")
    parser.add_argument(
        "--no_mask",
        action="store_true",
        help="Toggle to turn off masking. This can be used for prospectively undersampled data.",
    )
    parser.add_argument(
        "--mask_type",
        choices=("random", "gaussian2d", "equispaced"),
        default="gaussian2d",
        type=str,
        help="Type of k-space mask",
    )
    parser.add_argument(
        "--accelerations", nargs="+", default=[10, 10], type=int, help="Acceleration rates to use for masks"
    )
    parser.add_argument(
        "--center_fractions", nargs="+", default=[0.7, 0.7], type=float, help="Number of center lines to use in mask"
    )
    parser.add_argument("--shift_mask", action="store_true", help="Shift the mask")
    parser.add_argument("--normalize_inputs", action="store_true", help="Normalize the inputs")
    parser.add_argument("--crop_size", nargs="+", help="Size of the crop to apply to the input")
    parser.add_argument("--crop_before_masking", action="store_true", help="Crop before masking")
    parser.add_argument("--kspace_zero_filling_size", nargs="+", help="Size of zero-filling in kspace")
    parser.add_argument("--fft_type", type=str, default="orthogonal", help="Type of FFT to use")
    parser.add_argument("--progress_bar_refresh", type=int, default=10, help="Progress bar refresh rate")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for the data loader")
    parser.add_argument(
        "--data_parallel", action="store_true", help="If set, use multiple GPUs using data parallelism"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Which device to run on")

    return parser


if __name__ == "__main__":
    main(create_arg_parser().parse_args(sys.argv[1:]))
