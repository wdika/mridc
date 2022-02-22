# coding=utf-8
__author__ = "Dimitrios Karkalousos"

import argparse
import gc
import logging
import os
import pathlib
import random
import sys
import time
from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
import torchvision
from torch.nn import L1Loss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from mridc import SSIMLoss
from mridc.data.subsample import create_mask_for_mask_type
from mridc.data.transforms import PhysicsInformedDataTransform, center_crop_to_smallest
from mridc.evaluate import mse, psnr, nmse, ssim
from mridc.nn.e2evn import VarNet
from scripts.train_cirim import create_training_loaders, save_model, load_model, build_optim

torch.cuda.current_device()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def train_epoch(
    args: argparse.Namespace,
    epoch: int,
    iteration: int,
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    writer: torch.utils.tensorboard.SummaryWriter,
    best_dev_loss: float,
) -> Tuple[Union[float, Any], Union[float, Any], float, int]:
    """
    Train the model for one epoch.

    Args:
        args: Arguments for device selection.
        epoch: Current epoch.
        iteration: Current iteration.
        model: Model to train.
        train_loader: Data loader for the training set.
        val_loader: Data loader for the validation set.
        optimizer: Optimizer to use.
        loss_fn: Loss function to use.
        writer: Summary writer for Tensorboard.
        best_dev_loss: Best validation loss so far.

    Returns:
        Best average validation loss, validation loss, time taken, and iteration.
    """
    model.train()
    avg_loss = 0.0
    start_epoch = time.perf_counter()
    global_step = epoch * len(train_loader)

    memory_allocated = []
    for i, data in enumerate(train_loader):
        masked_kspace, sensitivity_map, mask, target, _, _, acc, max_value, _ = data
        sensitivity_map = sensitivity_map.to(args.device)
        target = target.to(args.device)
        max_value = max_value.to(args.device)

        if isinstance(masked_kspace, list):
            r = np.random.randint(len(masked_kspace))
            y = masked_kspace[r].to(args.device)
            m = mask[r].to(args.device)
            acceleration = str(acc[r].item())
        else:
            y = masked_kspace.to(args.device)
            m = mask.to(args.device)
            acceleration = str(acc.item())

        optimizer.zero_grad()
        model.zero_grad()
        output = model.forward(y, sensitivity_map, m)
        target, output = center_crop_to_smallest(target, output)

        loss = (
            loss_fn(output.unsqueeze(1), target.unsqueeze(1), data_range=max_value)
            if "ssim" in str(loss_fn).lower()
            else loss_fn(output, target)
        )

        loss.backward()
        optimizer.step()

        avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if i > 0 else loss.item()
        writer.add_scalar(f"Loss_{acceleration}x", loss.item(), global_step + i)
        writer.add_scalar("Total_Loss", avg_loss, global_step + i)

        if args.device == "cuda":
            memory_allocated.append(torch.cuda.max_memory_allocated() * 1e-6)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        gc.collect()

        if i % args.report_interval == 0:
            iteration += args.report_interval

            val_loss, mse_loss, nmse_loss, psnr_loss, ssim_loss, _ = evaluate(
                args, iteration, model, val_loader, loss_fn, writer
            )

            logging.info(
                f"Epoch = [{epoch:3d}/{args.num_epochs:3d}] "
                f"Iter = [{i:4d}/{len(train_loader):4d}] "
                f"Loss_{acceleration}x = {loss.detach().item():.4g} "
                f"Avg Loss = {avg_loss:.4g} "
            )

            _val_loss = []
            for acc in mse_loss:
                logging.info(
                    f"VAL_{str(loss_fn)}_{acc}x = {val_loss[acc]:.4g} "
                    f"VAL_MSE_{acc}x = {mse_loss[acc]:.4g} "
                    f"VAL_NMSE_{acc}x = {nmse_loss[acc]:.4g} "
                    f"VAL_PSNR_{acc}x = {psnr_loss[acc]:.4g} "
                    f"VAL_SSIM_{acc}x = {ssim_loss[acc]:.4g} "
                )
                _val_loss.append(val_loss[acc])

            _val_loss = sum(_val_loss) / len(_val_loss)  # type: ignore

            is_new_best = -_val_loss < best_dev_loss  # type: ignore
            best_dev_loss = min(best_dev_loss, _val_loss)  # type: ignore
            save_model(args, args.exp_dir, iteration, model, optimizer, best_dev_loss, is_new_best)

        if args.exit_after_checkpoint:
            writer.close()
            sys.exit(0)

        memory_allocated = []
    optimizer.zero_grad()

    return avg_loss, _val_loss, time.perf_counter() - start_epoch, iteration


def evaluate(
    args: argparse.Namespace,
    epoch: int,
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    writer: SummaryWriter,
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    float,
]:
    """
    Evaluate the model on the validation set.

    Args:
        args: Arguments as parsed by argparse.
        epoch: Current epoch.
        model: Model to evaluate.
        data_loader: Data loader for the validation set.
        loss_fn: Loss function to use.
        writer: Summary writer.

    Returns:
        Tuple of the following:
            - MSE loss.
            - NMSE loss.
            - PSNR loss.
            - SSIM loss.
            - Total loss.
            - Time taken.
    """
    model.eval()
    val_losses = {}
    mse_losses = {}
    psnr_losses = {}
    nmse_losses = {}
    ssim_losses = {}
    memory_allocated = []

    start = time.perf_counter()
    with torch.no_grad():
        for _, data in enumerate(data_loader):
            masked_kspace, sensitivity_map, mask, target, _, _, acc, max_value, _ = data
            sensitivity_map = sensitivity_map.to(args.device)
            target = target.to(args.device)
            max_value = max_value.to(args.device)

            if isinstance(masked_kspace, list):
                for _, r in enumerate(masked_kspace):
                    y = masked_kspace[r].to(args.device)
                    m = mask[r].to(args.device)
                    acceleration = str(acc[r].item())

                    output = model.forward(y, sensitivity_map, m)
                    target, output = center_crop_to_smallest(target, output)
                    output_np = output.cpu().numpy()
                    target_np = target.cpu().numpy()

                    if "ssim" in str(loss_fn).lower():
                        val_losses[acceleration] = (
                            loss_fn(target.unsqueeze(1), output.unsqueeze(1), data_range=max_value).cpu().numpy()
                        )
                    else:
                        val_losses[acceleration] = loss_fn(target.unsqueeze(1), output.unsqueeze(1)).cpu().numpy()

                    mse_losses[acceleration] = mse(target_np, output_np)
                    nmse_losses[acceleration] = nmse(target_np, output_np)
                    psnr_losses[acceleration] = psnr(target_np, output_np)
                    ssim_losses[acceleration] = ssim(target_np, output_np)
            else:
                y = masked_kspace.to(args.device)
                m = mask.to(args.device)
                acceleration = str(acc.item())

                output = model.inference(y, sensitivity_map, m)
                target, output = center_crop_to_smallest(target, output)
                output_np = output.cpu().numpy()
                target_np = target.cpu().numpy()

                if "ssim" in str(loss_fn).lower():
                    val_losses[acceleration] = (
                        loss_fn(target.unsqueeze(1), output.unsqueeze(1), data_range=max_value).cpu().numpy()
                    )
                else:
                    val_losses[acceleration] = loss_fn(target.unsqueeze(1), output.unsqueeze(1)).cpu().numpy()

                mse_losses[acceleration] = mse(target_np, output_np)
                nmse_losses[acceleration] = nmse(target_np, output_np)
                psnr_losses[acceleration] = psnr(target_np, output_np)
                ssim_losses[acceleration] = ssim(target_np, output_np)

            if args.device == "cuda":
                memory_allocated.append(torch.cuda.max_memory_allocated() * 1e-6)
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()

        val_loss = {}
        mse_loss = {}
        nmse_loss = {}
        psnr_loss = {}
        ssim_loss = {}
        for acc in mse_losses:
            val_loss[acc] = np.mean(val_losses[acc])
            mse_loss[acc] = np.mean(mse_losses[acc])
            nmse_loss[acc] = np.mean(nmse_losses[acc])
            psnr_loss[acc] = np.mean(psnr_losses[acc])
            ssim_loss[acc] = np.mean(ssim_losses[acc])

            writer.add_scalar(f"Val_{str(loss_fn)}_{acc}x", val_loss[acc], epoch)
            writer.add_scalar(f"Val_MSE_{acc}x", mse_loss[acc], epoch)
            writer.add_scalar(f"Val_NMSE_{acc}x", nmse_loss[acc], epoch)
            writer.add_scalar(f"Val_PSNR_{acc}x", psnr_loss[acc], epoch)
            writer.add_scalar(f"Val_SSIM_{acc}x", ssim_loss[acc], epoch)

    return (val_loss, mse_loss, nmse_loss, psnr_loss, ssim_loss, time.perf_counter() - start)


def visualize(
    args: argparse.Namespace,
    epoch: int,
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    writer: SummaryWriter,
):
    """
    Visualize the model's predictions.

    Args:
        args: Arguments as parsed by argparse.
        epoch: Current epoch.
        model: Model to visualize.
        data_loader: Data loader to use for visualization.
        writer: Summary writer to use for visualization.

    Returns:
        None
    """

    def save_image(image, tag):
        """
        Save image to tensorboard.

        Args:
            image: Image to save.
            tag: Tag to use for saving.

        Returns:
            None
        """
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)

    model.eval()
    with torch.no_grad():
        for _, data in enumerate(data_loader):
            masked_kspace, sensitivity_map, mask, target, _, _, acc, _, _ = data
            sensitivity_map = sensitivity_map.to(args.device)
            target = target.to(args.device)

            if isinstance(masked_kspace, list):
                r = np.random.randint(len(masked_kspace))
                y = masked_kspace[r].to(args.device)
                m = mask[r].to(args.device)
                acceleration = str(acc[r].item())
            else:
                y = masked_kspace.to(args.device)
                m = mask.to(args.device)
                acceleration = str(acc.item())

            output = model.forward(y, sensitivity_map, m)
            target, output = center_crop_to_smallest(target, output)
            output.detach()

    save_image(target, "Target")
    save_image(output, f"Reconstruction_{acceleration}x")
    save_image(target - output, f"Error_{acceleration}x")


def build_model(args):
    """
    Build the model.

    Args:
        args: Arguments as parsed by argparse.

    Returns:
        Model to use for training.
    """
    model = VarNet(
        num_cascades=args.num_cascades,  # number of unrolled iterations
        pools=args.pools,  # number of pooling layers for U-Net
        chans=args.chans,  # number of top-level channels for U-Net
        unet_padding_size=args.unet_padding_size,  # padding size for U-Net
        normalize=args.normalize,  # normalize input
        no_dc=args.no_dc,  # remove DC component
        use_sens_net=args.use_sens_net,  # use the sensitivity network
        sens_pools=args.sens_pools,  # number of pooling layers for sense est. U-Net
        sens_chans=args.sens_chans,  # number of top-level channels for sense est. U-Net
        sens_normalize=args.sens_normalize,  # normalize the sensitivity maps
        sens_mask_type=args.sens_mask_type,  # type of mask for sensitivity maps
        output_type=args.output_type,  # type of output
        fft_type=args.fft_type,  # type of FFT
    )

    return model.to(args.device)


def main(args):
    """
    Main function.

    Args:
        args: Arguments as parsed by argparse.

    Returns:
        None
    """
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    (args.exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=args.exp_dir / "summary")

    checkpoint_pretrained = os.path.join(args.exp_dir, "pretrained.pt")
    if args.checkpoint is None:
        checkpoint_path = os.path.join(args.exp_dir, "model.pt")
    else:
        checkpoint_path = args.checkpoint

    if args.resume and os.path.exists(checkpoint_path):
        checkpoint, model, optimizer = load_model(checkpoint_path)

        epochs = args.num_epochs
        args = checkpoint["args"]
        args.num_epochs = epochs

        best_dev_loss = checkpoint["best_dev_loss"]
        start_epoch = checkpoint["epoch"]  # + 1
    else:
        model = build_model(args)
        if args.data_parallel:
            model = torch.nn.DataParallel(model)
        optimizer = build_optim(args, model.parameters())
        if os.path.exists(checkpoint_pretrained):
            _, model, optimizer = load_model(checkpoint_pretrained)
            optimizer.lr = args.lr
        best_dev_loss = 1e9
        start_epoch = 0
    logging.info(args)
    logging.info(model)

    if args.accelerations[0] != args.accelerations[1] or len(args.accelerations) > 2:
        mask_func: list = []
        for _, acc in enumerate(args.accelerations):
            mask_func += create_mask_for_mask_type(args.mask_type, args.center_fractions[acc], args.accelerations[acc])
    else:
        mask_func = create_mask_for_mask_type(args.mask_type, args.center_fractions, args.accelerations)

    train_transform = PhysicsInformedDataTransform(
        mask_func=mask_func,
        shift_mask=args.shift_mask,
        normalize_inputs=args.normalize_inputs,
        crop_size=args.crop_size,
        crop_before_masking=args.crop_before_masking,
        fft_type=args.fft_type,
        use_seed=True,
    )
    val_transform = PhysicsInformedDataTransform(
        mask_func=mask_func,
        shift_mask=args.shift_mask,
        normalize_inputs=args.normalize_inputs,
        crop_size=args.crop_size,
        crop_before_masking=args.crop_before_masking,
        fft_type=args.fft_type,
    )

    train_loader, val_loader, display_loader = create_training_loaders(args, train_transform, val_transform)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)

    if args.loss_fn == "ssim":
        val_loss_fn = SSIMLoss()
        val_loss_fn = val_loss_fn.to(args.device)
    else:
        val_loss_fn = L1Loss()

    # This is needed for resuming training.
    iteration = start_epoch
    start_epoch = int(np.floor(iteration / len(train_loader)))
    for epoch in range(start_epoch, args.num_epochs):
        _, val_loss, _, iteration = train_epoch(
            args,
            epoch,
            iteration,
            model,
            train_loader,
            val_loader,
            optimizer,
            val_loss_fn,
            writer,
            best_dev_loss=best_dev_loss,
        )
        visualize(args, epoch, model, display_loader, writer)
        scheduler.step()

        best_dev_loss = min(best_dev_loss, val_loss)

        if args.exit_after_checkpoint:
            writer.close()
            sys.exit(0)

    writer.close()


def create_arg_parser():
    """
    Creates an ArgumentParser to read the arguments.

    Returns:
        argparse.ArgumentParser: An ArgumentParser object.
    """
    parser = argparse.ArgumentParser(description="E2EVN")

    parser.add_argument("data_path", type=pathlib.Path, help="Path to the data folder")
    parser.add_argument(
        "exp_dir", type=pathlib.Path, default="checkpoints", help="Path where model and results should be saved"
    )
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
    parser.add_argument("--crop_size", default=None, help="Size of the crop to apply to the input")
    parser.add_argument("--crop_before_masking", action="store_true", help="Crop before masking")
    parser.add_argument("--num_cascades", type=int, default=12, help="Number of cascades for the model")
    parser.add_argument("--pools", type=int, default=2, help="Number of pools for the model")
    parser.add_argument("--chans", type=int, default=12, help="Number of channels for the model")
    parser.add_argument("--unet_padding_size", type=int, default=11, help="Padding size for the unet")
    parser.add_argument("--normalize", action="store_true", help="Normalize the inputs")
    parser.add_argument("--no_dc", action="store_true", help="Do not use DC component")
    parser.add_argument("--use_sens_net", action="store_true", help="Use sensitivity net")
    parser.add_argument("--sens_pools", type=int, default=4, help="Number of pools for the sensitivity net")
    parser.add_argument("--sens_chans", type=int, default=8, help="Number of channels for the sensitivity net")
    parser.add_argument("--sens_normalize", action="store_true", help="Normalize the sensitivity net")
    parser.add_argument(
        "--sens_mask_type", choices=["1D", "2D"], default="2D", help="Type of mask to use for the sensitivity net"
    )
    parser.add_argument("--output_type", choices=["SENSE", "RSS"], default="SENSE", help="Type of output to use")
    parser.add_argument("--fft_type", type=str, default="orthogonal", help="Type of FFT to use")
    parser.add_argument("--batch_size", default=1, type=int, help="Mini batch size")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument(
        "--optimizer", type=str, default="Adam", help="Optimizer to use choose between" "['Adam', 'SGD', 'RMSProp']"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--lr_step_size", type=int, default=40, help="Period of learning rate decay")
    parser.add_argument("--lr_gamma", type=float, default=0.1, help="Multiplicative factor of learning rate decay")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Strength of weight decay regularization")
    parser.add_argument("--report_interval", type=int, default=150, help="Period of loss reporting")
    parser.add_argument(
        "--resume",
        action="store_true",
        help='If set, resume the training from a previous model checkpoint. "--checkpoint" should be set with this',
    )
    parser.add_argument("--checkpoint", type=str, help='Path to an existing checkpoint. Used along with "--resume"')
    parser.add_argument("--exit_after_checkpoint", action="store_true", help="If set, exit after loading a checkpoint")
    parser.add_argument("--seed", default=42, type=int, help="Seed for random number generators")
    parser.add_argument(
        "--data_parallel", action="store_true", help="If set, use multiple GPUs using data parallelism"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Which device to run on")

    return parser


if __name__ == "__main__":
    ARGS = create_arg_parser().parse_args()
    random.seed(ARGS.seed)
    np.random.seed(ARGS.seed)
    torch.manual_seed(ARGS.seed)
    main(ARGS)
