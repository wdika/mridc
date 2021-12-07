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
import torch.multiprocessing
from torch.utils.data import DataLoader

from mridc import save_reconstructions
from mridc.data.mri_data import SliceDataset
from mridc.data.subsample import create_mask_for_mask_type
from mridc.data.transforms import PhysicsInformedDataTransform, center_crop_to_smallest
from mridc.nn.cirim import CIRIM
from scripts.train_cirim import build_optim

torch.multiprocessing.set_sharing_strategy("file_system")
torch.backends.cudnn.benchmark = False


def load_model(
    checkpoint_file: str, fft_type: str, device: str
) -> Tuple[Any, Union[torch.nn.DataParallel, CIRIM], torch.optim.Optimizer]:
    """
    Loads the model from the checkpoint file.

    Args:
        checkpoint_file: Path to the checkpoint file.
        fft_type: Type of the FFT.
        device: Device to use.

    Returns:
        Checkpoint, CIRIM model, optimizer.
    """
    checkpoint = torch.load(checkpoint_file, map_location=torch.device("cpu"))
    arguments = checkpoint["args"]

    model = CIRIM(
        num_cascades=arguments.num_cascades,  # number of unrolled iterations
        time_steps=arguments.time_steps,  # number of time steps
        recurrent_layer=arguments.recurrent_layer,  # recurrent layer type
        conv_filters=arguments.conv_filters,  # number of filters in each conv layer
        conv_kernels=arguments.conv_kernels,  # size of kernel in each conv layer
        conv_dilations=arguments.conv_dilations,  # dilation in each conv layer
        conv_bias=arguments.conv_bias,  # bias in each conv layer
        recurrent_filters=arguments.recurrent_filters,  # number of filters in each recurrent layer
        recurrent_kernels=arguments.recurrent_kernels,  # size of kernel in each recurrent layer
        recurrent_dilations=arguments.recurrent_dilations,  # dilation in each recurrent layer
        recurrent_bias=arguments.recurrent_bias,  # bias in each recurrent layer
        depth=arguments.depth,  # number of cascades
        conv_dim=arguments.conv_dim,  # dimensionality of input
        no_dc=arguments.no_dc,  # turn on/off DC component
        keep_eta=arguments.keep_eta,  # keep the eta signal
        use_sens_net=arguments.use_sens_net,  # use the sensitivity network
        sens_pools=arguments.sens_pools,  # number of pooling layers for sense est. U-Net
        sens_chans=arguments.sens_chans,  # number of top-level channels for sense est. U-Net
        sens_normalize=arguments.sens_normalize,  # normalize the sensitivity maps
        sens_mask_type=arguments.sens_mask_type,  # type of mask for sensitivity maps
        output_type=arguments.output_type,  # type of output
        fft_type=fft_type,  # type of FFT
    ).to(device)

    if "loss_fn.w" in checkpoint["model"]:
        del checkpoint["model"]["loss_fn.w"]

    if arguments.data_parallel:
        model = torch.nn.DataParallel(model)  # type: ignore

    model.load_state_dict(checkpoint["model"])

    optimizer = build_optim(arguments, model.parameters())  # type: ignore
    optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint, model, optimizer


def run_cirim(model: CIRIM, data_loader: DataLoader, device: str, progress_bar_refresh: int) -> Dict[str, np.ndarray]:
    """
    Runs the reconstruction.

    Args:
        model: Cascades of Independently Recurrent Inference Machines
        data_loader: torch.utils.data.DataLoader
        device: cuda or cpu
        progress_bar_refresh: Refresh rate of the progress bar.

    Returns
    -------
        Dictionary with the reconstructions.
    """
    # set the model to evaluation mode
    model.eval()
    model.to(device)

    # initialize the output dictionary
    output = defaultdict(list)

    # loop over the data loader
    for i, data in enumerate(data_loader):
        # move the data to the correct device
        (masked_kspace, sensitivity_map, mask, eta, target, fname, slice_num, _, _, _) = data

        sensitivity_map = sensitivity_map.to(device)
        target = target.to(device)

        if eta is not None and eta.size != 0:
            eta = eta.to(device)

        y = masked_kspace.to(device)
        m = mask.to(device)

        # run the forward pass
        with torch.no_grad():
            try:
                estimate = next(model.inference(y, sensitivity_map, m, eta=eta, accumulate_estimates=True))
            except StopIteration:
                break

            if isinstance(estimate, list):
                estimate = estimate[0][-1]

                if estimate.shape[-1] == 2:
                    estimate = torch.view_as_complex(estimate)

        estimate = estimate / torch.max(torch.abs(estimate))
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
            transform=PhysicsInformedDataTransform(
                mask_func=create_mask_for_mask_type(args.mask_type, args.center_fractions, args.accelerations),
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

    # load the model
    _, model, _ = load_model(args.checkpoint, args.fft_type, args.device)

    init_start = time.perf_counter()

    # TODO: change print to logger
    print("Reconstructing...")
    reconstructions = run_cirim(model, data_loader, args.device, args.progress_bar_refresh)

    print("Saving...")
    save_reconstructions(reconstructions, args.out_dir)

    print("Finished! It took", time.perf_counter() - init_start, "s \n")


def create_arg_parser():
    """
    Creates and returns the ArgumentParser.

    Returns:
        ArgumentParser: An ArgumentParser object.
    """
    parser = argparse.ArgumentParser(description="CIRIM")

    parser.add_argument("data_path", type=pathlib.Path, help="Path to the data folder")
    parser.add_argument("checkpoint", type=pathlib.Path, help="Path to the checkpoint file")
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
