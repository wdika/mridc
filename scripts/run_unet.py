# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

import argparse
import gc
import pathlib
import sys
import time
from collections import defaultdict
from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from mridc import save_reconstructions, rss
from mridc.data.mri_data import SliceDataset
from mridc.data.subsample import create_mask_for_mask_type
from mridc.data.transforms import center_crop_to_smallest, UnetDataTransform
from mridc.nn.e2evn import NormUnet
from scripts.train_cirim import build_optim

torch.backends.cudnn.benchmark = False


def load_model(
    checkpoint_file: str, device: str
) -> Tuple[Any, Union[torch.nn.DataParallel, NormUnet], torch.optim.Optimizer]:
    """
    Loads the model from the checkpoint file.

    Args:
        checkpoint_file: Path to the checkpoint file.
        device: cuda or cpu

    Returns:
        Checkpoint, UNet model, optimizer.
    """
    checkpoint = torch.load(checkpoint_file, map_location=device)
    arguments = checkpoint["args"]

    model = NormUnet(
        in_chans=arguments.in_chans,  # number of channels in input image
        out_chans=arguments.out_chans,  # number of channels in output image
        chans=arguments.chans,  # number of channels in intermediate layers
        # number of pooling operations in the encoder/decoder
        num_pools=arguments.num_pools,
        drop_prob=arguments.drop_prob,  # dropout probability
        padding_size=arguments.padding_size,  # padding size
        normalize=arguments.normalize,  # normalize the input image
    ).to(device)

    if arguments.data_parallel:
        model = torch.nn.DataParallel(model)  # type: ignore

    model.load_state_dict(checkpoint["model"])

    optimizer = build_optim(arguments, model.parameters())  # type: ignore
    optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint, model, optimizer


def run_unet(
    model: NormUnet, data_loader: DataLoader, output_type: str, device: str, progress_bar_refresh: int
) -> Dict[str, np.ndarray]:
    """
    Runs the model on the data loader and returns the reconstructions.

    Args:
        model: Normalized Unet
        data_loader: torch.utils.data.DataLoader
        output_type: SENSE or RSS
        device: cuda or cpu
        progress_bar_refresh: Refresh rate of the progress bar.

    Returns:
        Dictionary with the reconstructions.
    """
    model.eval()
    model.to(device)

    # Create a dictionary to store the results
    output = defaultdict(list)
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            image, target, fname, slice_num, _, _, _ = data

            # Move the data to the correct device
            image = image.to(device)
            target = target.to(device)

            # Run the model
            estimate = model.forward(image.unsqueeze(1)).squeeze(1)

            if output_type == "SENSE":
                estimate = torch.view_as_complex(estimate)
            elif output_type == "rss":
                estimate = rss(estimate, dim=1)
            else:
                raise ValueError(f"Unknown output_type: {output_type}")

            target, estimate = center_crop_to_smallest(target, estimate)
            output[fname[0]].append((slice_num, estimate.cpu()))

            gc.collect()
            torch.cuda.empty_cache()

            # update the progress bar
            if i % progress_bar_refresh == 0:
                sys.stdout.write("\r[{:5.2f}%]".format(
                    100 * (i + 1) / len(data_loader)))
                sys.stdout.flush()

    sys.stdout.write("\n")
    sys.stdout.flush()

    reconstructions = {
        fname: np.stack([pred for _, pred in sorted(slice_preds)]) for fname, slice_preds in output.items()
    }

    return reconstructions


def main(args):
    """
    Main function.

    Args:
        args: Arguments from the command line.

    Returns:
        None
    """
    data_loader = DataLoader(
        dataset=SliceDataset(
            root=args.data_path,
            sense_root=args.sense_path,
            challenge=args.challenge,
            transform=UnetDataTransform(
                mask_func=False
                if args.no_mask
                else create_mask_for_mask_type(args.mask_type, args.center_fractions, args.accelerations),
                shift_mask=args.shift_mask,
                normalize_inputs=args.normalize_inputs,
                crop_size=args.crop_size,
                crop_before_masking=args.crop_before_masking,
                kspace_zero_filling_size=args.kspace_zero_filling_size,
                fft_type=args.fft_type,
                output_type=args.output_type,
                use_seed=False,
            ),
            sample_rate=args.sample_rate,
            mask_root=args.mask_path,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    # load the model
    _, model, _ = load_model(args.checkpoint, args.device)

    init_start = time.perf_counter()

    # TODO: change print to logger
    print("Reconstructing...")
    reconstructions = run_unet(
        model, data_loader, args.output_type, args.device, args.progress_bar_refresh)

    print("Saving...")
    save_reconstructions(reconstructions, args.out_dir)

    print("Finished! It took", time.perf_counter() - init_start, "s \n")


def create_arg_parser():
    """
    Creates an ArgumentParser to read the arguments.

    Returns:
        argparse.ArgumentParser: An ArgumentParser object.
    """
    parser = argparse.ArgumentParser(description="UNET")

    parser.add_argument("data_path", type=pathlib.Path,
                        help="Path to the data folder")
    parser.add_argument("checkpoint", type=pathlib.Path,
                        help="Path to the checkpoint file")
    parser.add_argument("out_dir", type=pathlib.Path,
                        help="Path to the output folder")
    parser.add_argument("--sense_path", type=pathlib.Path,
                        help="Path to the sense folder")
    parser.add_argument("--mask_path", type=pathlib.Path,
                        help="Path to the mask folder")
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
    parser.add_argument("--sample_rate", type=float,
                        default=1.0, help="Sample rate for the data")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for the data loader")
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
    parser.add_argument("--shift_mask", action="store_true",
                        help="Shift the mask")
    parser.add_argument("--normalize_inputs",
                        action="store_true", help="Normalize the inputs")
    parser.add_argument("--crop_size", nargs="+",
                        help="Size of the crop to apply to the input")
    parser.add_argument("--crop_before_masking",
                        action="store_true", help="Crop before masking")
    parser.add_argument("--kspace_zero_filling_size",
                        nargs="+", help="Size of zero-filling in kspace")
    parser.add_argument(
        "--output_type", choices=("SENSE", "RSS"), default="SENSE", type=str, help="Type of output to save"
    )
    parser.add_argument("--fft_type", type=str,
                        default="orthogonal", help="Type of FFT to use")
    parser.add_argument("--progress_bar_refresh", type=int,
                        default=10, help="Progress bar refresh rate")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for the data loader")
    parser.add_argument(
        "--data_parallel", action="store_true", help="If set, use multiple GPUs using data parallelism"
    )
    parser.add_argument("--device", type=str, default="cuda",
                        help="Which device to run on")

    return parser


if __name__ == "__main__":
    main(create_arg_parser().parse_args(sys.argv[1:]))
