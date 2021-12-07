# coding=utf-8
__author__ = "Dimitrios Karkalousos"

import argparse
import logging
import multiprocessing
import pathlib
import time
from collections import defaultdict
from typing import Any, List, Optional, Tuple, Union

import bart
import numpy as np
import torch
import torch.nn.functional as F

from mridc import ifft2c, fft2c, save_reconstructions
from mridc.data.mri_data import SliceDataset
from mridc.data.subsample import create_mask_for_mask_type, MaskFunc
from mridc.data.transforms import (
    to_tensor,
    complex_center_crop,
    center_crop_to_smallest,
    apply_mask,
    tensor_to_complex_np,
)


def save_outputs(outputs: List, output_path: pathlib.Path):
    """
    Save outputs to disk.

    Args:
        outputs: List of outputs.
        output_path: Path to save outputs to.

    Returns:
        None
    """
    reconstructions = defaultdict(list)

    for i, _ in enumerate(outputs):
        fname, slice_num, pred = outputs[i]
        reconstructions[fname].append((slice_num, pred))

    reconstructions = {
        fname: np.stack([pred for _, pred in sorted(slice_preds)]) for fname, slice_preds in reconstructions.items()
    }  # type: ignore

    save_reconstructions(reconstructions, output_path / "reconstructions")


class PICSDataTransform:
    """Data transform for PICS."""

    def __init__(
        self,
        mask_func: MaskFunc = None,
        shift_mask: bool = False,
        crop_size: Optional[Union[Tuple, None]] = None,
        kspace_crop: bool = False,
        crop_before_masking: bool = True,
        kspace_zero_filling_size: Optional[Tuple] = None,
        fft_type: str = "data",
        use_seed: bool = True,
    ):
        """
        Args:
            mask_func: Function to create mask.
            shift_mask: Whether to shift mask.
            crop_size: Size of crop.
            kspace_crop: Whether to crop kspace.
            crop_before_masking: Whether to crop before masking.
            kspace_zero_filling_size: The size of padding in kspace -> zero filling.
            fft_type: Type of fft to use.
            use_seed: Whether to use seed.
        """
        self.mask_func = mask_func
        self.shift_mask = shift_mask

        self.crop_size = crop_size
        self.crop_before_masking = crop_before_masking
        self.kspace_zero_filling_size = kspace_zero_filling_size
        self.kspace_crop = kspace_crop

        self.fft_type = fft_type

        self.use_seed = use_seed

    def __call__(
        self,
        kspace: np.ndarray,
        sensitivity_map: np.ndarray,
        mask: np.ndarray,
        eta: np.ndarray,
        target: np.ndarray,
        attrs: dict,
        fname: str,
        slice_num: int,
    ) -> Tuple[Union[torch.Tensor, Any], Optional[torch.Tensor], np.ndarray, str, int]:
        """
        Args:
            kspace: Fully sampled kspace. If kspace is prospectively undersampled, then self.mask_func should be None.
            sensitivity_map: Coil sensitivities map.
            mask: Undersampling mask.
            eta: Initial estimation. It loads from the mri_data, but won't be used here.
            target: Target. It loads from the mri_data and used for checking cropping size.
            attrs: Attributes. It loads from the mri_data and used for checking cropping size.
            fname: File name.
            slice_num: Slice number.

        Returns:
            Tuple of (kspace, mask, eta, target, attrs, fname, slice_num)
        """
        if self.fft_type != "data":
            sensitivity_map = np.fft.fftshift(sensitivity_map, axes=(-2, -1))

        kspace = to_tensor(kspace)
        sensitivity_map = to_tensor(sensitivity_map)

        # Apply zero-filling on kspace
        if self.kspace_zero_filling_size is not None and self.kspace_zero_filling_size != "":
            # (padding_left,padding_right, padding_top,padding_bottom)
            padding_top = abs(int(self.kspace_zero_filling_size[0]) - kspace.shape[1]) // 2
            padding_bottom = padding_top
            padding_left = abs(int(self.kspace_zero_filling_size[1]) - kspace.shape[2]) // 2
            padding_right = padding_left

            kspace = torch.view_as_complex(kspace)
            kspace = F.pad(
                kspace, pad=(padding_left, padding_right, padding_top, padding_bottom), mode="constant", value=0
            )
            kspace = torch.view_as_real(kspace)

            sensitivity_map = fft2c(sensitivity_map, self.fft_type)
            sensitivity_map = torch.view_as_complex(sensitivity_map)
            sensitivity_map = F.pad(
                sensitivity_map,
                pad=(padding_left, padding_right, padding_top, padding_bottom),
                mode="constant",
                value=0,
            )
            sensitivity_map = torch.view_as_real(sensitivity_map)
            sensitivity_map = ifft2c(sensitivity_map, self.fft_type)

        crop_size = torch.tensor([attrs["recon_size"][0], attrs["recon_size"][1]])

        if self.crop_size is not None:
            # Check for smallest size against the target shape.
            h = self.crop_size[0] if self.crop_size[0] <= target.shape[0] else target.shape[0]
            w = self.crop_size[1] if self.crop_size[1] <= target.shape[1] else target.shape[1]

            # Check for smallest size against the stored recon shape in metadata.
            if crop_size[0] != 0:
                h = h if h <= crop_size[0] else crop_size[0]
            if crop_size[1] != 0:
                w = w if w <= crop_size[1] else crop_size[1]

            self.crop_size = (h, w)
            # crop_size = torch.tensor([self.crop_size[0], self.crop_size[1]])

            if sensitivity_map is not None and sensitivity_map.size != 0:
                sensitivity_map = complex_center_crop(sensitivity_map, self.crop_size)

        # Cropping before masking will maintain the shape of original kspace intact for masking.
        if self.crop_size is not None and self.crop_before_masking:
            kspace = (
                complex_center_crop(kspace, self.crop_size)
                if self.kspace_crop
                else fft2c(
                    complex_center_crop(ifft2c(kspace, fft_type=self.fft_type), self.crop_size), fft_type=self.fft_type
                )
            )

        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, _, _ = apply_mask(kspace, self.mask_func, seed, (0,), shift=self.shift_mask)
        else:
            masked_kspace = kspace

        # Cropping after masking.
        if self.crop_size is not None and not self.crop_before_masking:
            masked_kspace = (
                complex_center_crop(masked_kspace, self.crop_size)
                if self.kspace_crop
                else fft2c(
                    complex_center_crop(ifft2c(masked_kspace, fft_type=self.fft_type), self.crop_size),
                    fft_type=self.fft_type,
                )
            )

        masked_kspace = tensor_to_complex_np(masked_kspace.permute(1, 2, 0, 3).unsqueeze(0).cpu())
        sensitivity_map = tensor_to_complex_np(sensitivity_map.permute(1, 2, 0, 3).unsqueeze(0).cpu())

        return masked_kspace, sensitivity_map, target, fname, slice_num


def cs_total_variation(
    arguments: argparse.Namespace, masked_kspace: np.ndarray, sensitivity_map: np.ndarray, target: np.ndarray
) -> np.ndarray:
    """
    Compute PICS reconstruction using total variation regularization.

    Args:
        arguments: Arguments.
        masked_kspace: Masked kspace.
        sensitivity_map: Sensitivity map.
        target: Target.

    Returns:
        Reconstructed image.
    """
    if arguments.device == "cuda":
        pred = bart.bart(
            1, f"pics -d0 -g -S -R W:7:0:{arguments.reg_wt} -i {arguments.num_iters}", masked_kspace, sensitivity_map
        )[0]
    else:
        pred = bart.bart(
            1, f"pics -d0 -S -R W:7:0:{arguments.reg_wt} -i {arguments.num_iters}", masked_kspace, sensitivity_map
        )[0]

    target, pred = center_crop_to_smallest(target, pred)

    return pred / np.max(pred)


def run_pics(idx):
    """
    Run PICS reconstruction.

    Args:
        idx: Index of the image.

    Returns:
        Reconstructed image.
    """
    masked_kspace, sensitivity_map, target, fname, slice_num = dataset[idx]
    start_time = time.perf_counter()

    logging.info(f"{fname} {slice_num}")
    prediction = cs_total_variation(ARGS, masked_kspace, sensitivity_map, target)
    logging.info(f"Done in {time.perf_counter() - start_time:.4f}s")

    return fname, slice_num, prediction


def main(args):
    """
    Main function.

    Args:
        args: Arguments.

    Returns:
        None.
    """
    if args.num_procs != 0:
        # Run multiprocessing.
        with multiprocessing.Pool(args.num_procs) as pool:
            start_time = time.perf_counter()
            outputs = pool.map(run_pics, range(len(dataset)))
            time_taken = time.perf_counter() - start_time
    else:
        start_time = time.perf_counter()
        outputs = []
        for i in range(len(dataset)):
            outputs.append(run_pics(i))
        time_taken = time.perf_counter() - start_time

    logging.info(f"Run Time = {time_taken:} s")
    save_outputs(outputs, args.out_dir)


def create_arg_parser():
    """
    Creates an ArgumentParser to read the arguments.

    Returns:
        argparse.ArgumentParser: An ArgumentParser object.
    """
    parser = argparse.ArgumentParser(description="PICS")

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
    parser.add_argument(
        "--num_iters", type=int, default=60, help="Number of iterations to run the reconstruction algorithm"
    )
    parser.add_argument("--reg_wt", type=float, default=0.005, help="Regularization weight parameter")
    parser.add_argument(
        "--num_procs", type=int, default=16, help="Number of processes. Set to 0 to disable multiprocessing."
    )
    parser.add_argument("--fft_type", type=str, default="orthogonal", help="Type of FFT to use")
    parser.add_argument("--progress_bar_refresh", type=int, default=10, help="Progress bar refresh rate")
    parser.add_argument(
        "--data_parallel", action="store_true", help="If set, use multiple GPUs using data parallelism"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Which device to run on")

    return parser


if __name__ == "__main__":
    ARGS = create_arg_parser().parse_args()

    ARGS.crop_size = (
        [int(ARGS.crop_size[0]), int(ARGS.crop_size[1])]
        if ARGS.crop_size is not None and ARGS.crop_size != "None"
        else None
    )

    dataset = SliceDataset(
        root=ARGS.data_path,
        sense_root=ARGS.sense_path,
        transform=PICSDataTransform(
            mask_func=create_mask_for_mask_type(ARGS.mask_type, ARGS.center_fractions, ARGS.accelerations),
            shift_mask=ARGS.shift_mask,
            crop_size=ARGS.crop_size,  # type: ignore
            crop_before_masking=ARGS.crop_before_masking,
            kspace_zero_filling_size=ARGS.kspace_zero_filling_size,
            fft_type=ARGS.fft_type,
        ),
        challenge=ARGS.challenge,
        sample_rate=ARGS.sample_rate,
        use_dataset_cache=ARGS.use_dataset_cache,
    )

    main(ARGS)
