# coding=utf-8
__author__ = "Dimitrios Karkalousos"

import argparse
import gc
import logging
import os
import pathlib
import random
import shutil
import sys
import time
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torchvision
from torch.nn import L1Loss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from mridc import SSIMLoss
from mridc.data.mri_data import SliceDataset
from mridc.data.subsample import create_mask_for_mask_type
from mridc.data.transforms import PhysicsInformedDataTransform, center_crop_to_smallest
from mridc.evaluate import mse, psnr, nmse, ssim
from mridc.nn.cirim import CIRIM

torch.cuda.current_device()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def create_training_loaders(
    args: argparse.Namespace,
    TrainingTransform: torchvision.transforms.Compose,
    ValTransform: torchvision.transforms.Compose,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create the training and validation data loaders.

    Args:
        args: Arguments for device selection.
        TrainingTransform: Training data transform.
        ValTransform: Validation data transform.

    Returns:
        Training data loader, validation data loader, and validation data loader.
    """
    train_loader = DataLoader(
        dataset=SliceDataset(
            root=args.data_path / f"{args.challenge}_train",
            sense_root=args.sense_path,
            challenge=args.challenge,
            transform=TrainingTransform,
            sample_rate=args.sample_rate,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    val_loader = DataLoader(
        dataset=SliceDataset(
            root=args.data_path / f"{args.challenge}_val",
            sense_root=args.sense_path,
            challenge=args.challenge,
            transform=ValTransform,
            sample_rate=args.sample_rate,
        ),
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    display_loader = DataLoader(
        dataset=SliceDataset(
            root=args.data_path / f"{args.challenge}_val",
            sense_root=args.sense_path,
            challenge=args.challenge,
            transform=ValTransform,
            sample_rate=args.sample_rate,
        ),
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    return train_loader, val_loader, display_loader


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
        train_loader: Training data loader.
        val_loader: Validation data loader.
        optimizer: Optimizer to use.
        loss_fn: Loss function to use.
        writer: Tensorboard writer.
        best_dev_loss: Best validation loss.

    Returns:
        Best average validation loss, validation loss, time taken, and iteration.
    """
    model.train()

    avg_loss = 0.0
    start_epoch = time.perf_counter()
    global_step = epoch * len(train_loader)

    memory_allocated = []
    for i, data in enumerate(train_loader):
        (masked_kspace, sensitivity_map, mask, eta,
         target, _, _, acc, max_value, _) = data
        sensitivity_map = sensitivity_map.to(args.device)
        target = target.to(args.device)
        max_value = max_value.to(args.device)

        if eta is not None and eta.size != 0:
            eta = eta.to(args.device)

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

        loss: torch.Tensor = sum(
            model.forward(y, sensitivity_map, m, eta=eta, target=target,
                          max_value=max_value, accumulate_loss=True)
        )

        loss.backward()  # type: ignore
        optimizer.step()

        avg_loss = 0.99 * avg_loss + 0.01 * \
            loss.item() if i > 0 else loss.item()  # type: ignore

        writer.add_scalar(f"Loss_{acceleration}x",
                          loss.item(), global_step + i)  # type: ignore
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
                f"Epoch = [{epoch:3d}/{args.num_epochs:3d}] "  # type: ignore
                f"Iter = [{i:4d}/{len(train_loader):4d}] "
                # type: ignore
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
            save_model(args, args.exp_dir, iteration, model,
                       optimizer, best_dev_loss, is_new_best)

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
            (masked_kspace, sensitivity_map, mask, eta,
             target, _, _, acc, max_value, _) = data
            sensitivity_map = sensitivity_map.to(args.device)
            target = target.to(args.device)

            if eta is not None and eta.size != 0:
                eta = eta.to(args.device)

            if isinstance(masked_kspace, list):
                for r, _ in enumerate(masked_kspace):
                    y = masked_kspace[r].to(args.device)
                    m = mask[r].to(args.device)
                    acceleration = str(acc[r].item())

                    try:
                        output = next(model.inference(
                            y, sensitivity_map, m, eta=eta, accumulate_estimates=True))
                    except StopIteration:
                        break

                    if isinstance(output, list):
                        output = output[0][-1]
                        output = output[..., 0] + 1j * output[..., 1]
                        output = torch.abs(
                            output / torch.max(torch.abs(output)))

                    target, output = center_crop_to_smallest(target, output)
                    output_np = output.cpu().numpy()
                    target_np = target.cpu().numpy()

                    if "ssim" in str(loss_fn).lower():
                        val_losses[acceleration] = (
                            loss_fn(target.unsqueeze(1), output.unsqueeze(
                                1), data_range=max_value.to(args.device))
                            .cpu()
                            .numpy()
                        )
                    else:
                        val_losses[acceleration] = loss_fn(
                            target.unsqueeze(1), output.unsqueeze(1)).cpu().numpy()

                    mse_losses[acceleration] = mse(target_np, output_np)
                    nmse_losses[acceleration] = nmse(target_np, output_np)
                    psnr_losses[acceleration] = psnr(target_np, output_np)
                    ssim_losses[acceleration] = ssim(target_np, output_np)
            else:
                y = masked_kspace.to(args.device)
                m = mask.to(args.device)
                acceleration = str(acc.item())

                try:
                    output = next(model.inference(
                        y, sensitivity_map, m, accumulate_estimates=True))
                except StopIteration:
                    break

                if isinstance(output, list):
                    output = output[0][-1]
                    output = output[..., 0] + 1j * output[..., 1]
                    output = torch.abs(output / torch.max(torch.abs(output)))

                target, output = center_crop_to_smallest(target, output)
                output_np = output.cpu().numpy()
                target_np = target.cpu().numpy()

                if "ssim" in str(loss_fn).lower():
                    val_losses[acceleration] = (
                        loss_fn(target.unsqueeze(1), output.unsqueeze(
                            1), data_range=max_value.to(args.device))
                        .cpu()
                        .numpy()
                    )
                else:
                    val_losses[acceleration] = loss_fn(
                        target.unsqueeze(1), output.unsqueeze(1)).cpu().numpy()

                mse_losses[acceleration] = mse(target_np, output_np)
                nmse_losses[acceleration] = nmse(target_np, output_np)
                psnr_losses[acceleration] = psnr(target_np, output_np)
                ssim_losses[acceleration] = ssim(target_np, output_np)

            if args.device == "cuda":
                memory_allocated.append(
                    torch.cuda.max_memory_allocated() * 1e-6)
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
            gc.collect()

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

            writer.add_scalar(
                f"Val_{str(loss_fn)}_{acc}x", val_loss[acc], epoch)
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
            (masked_kspace, sensitivity_map, mask,
             eta, target, _, _, acc, _, _) = data
            sensitivity_map = sensitivity_map.to(args.device)
            target = target.to(args.device)

            if eta is not None and eta.size != 0:
                eta = eta.to(args.device)

            if isinstance(masked_kspace, list):
                r = np.random.randint(len(masked_kspace))
                y = masked_kspace[r].to(args.device)
                m = mask[r].to(args.device)
                acceleration = str(acc[r].item())
            else:
                y = masked_kspace.to(args.device)
                m = mask.to(args.device)
                acceleration = str(acc.item())

            try:
                output = next(model.inference(y, sensitivity_map,
                              m, eta=eta, accumulate_estimates=True))
            except StopIteration:
                break

            if isinstance(output, list):
                output = output[0][-1]
                output = output[..., 0] + 1j * output[..., 1]
                output = torch.abs(output / torch.max(torch.abs(output)))

            target, output = center_crop_to_smallest(target, output)
            output.detach()

    save_image(target, "Target")
    save_image(output, f"Reconstruction_{acceleration}x")
    save_image(target - output, f"Error_{acceleration}x")


def save_model(
    args: argparse.Namespace,
    exp_dir: str,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    best_dev_loss: float,
    is_new_best: bool,
) -> None:
    """
    Save the model to disk.

    Args:
        args: Arguments as parsed by argparse.
        exp_dir: Directory to save the model to.
        epoch: Current epoch.
        model: Model to save.
        optimizer: Optimizer to save.
        best_dev_loss: Best validation loss.
        is_new_best: Whether the current validation loss is the best.

    Returns:
        None
    """
    torch.save(
        {
            "epoch": epoch,
            "args": args,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_dev_loss": best_dev_loss,
            "exp_dir": exp_dir,
        },
        f=pathlib.Path(
            str(exp_dir) + "/checkpoints/checkpoint_" + str(epoch) + ".pt"),
    )

    if is_new_best:
        shutil.copyfile(
            pathlib.Path(str(exp_dir) +
                         "/checkpoints/checkpoint_" + str(epoch) + ".pt"),
            pathlib.Path(exp_dir) / "best_model.pt",
        )


def build_model(args):
    """
    Build the model.

    Args:
        args: Arguments as parsed by argparse.

    Returns:
        Model to use for training.
    """
    model = CIRIM(
        num_cascades=args.num_cascades,  # number of unrolled iterations
        time_steps=args.time_steps,  # number of time steps
        recurrent_layer=args.recurrent_layer,  # recurrent layer type
        conv_filters=args.conv_filters,  # number of filters in each conv layer
        conv_kernels=args.conv_kernels,  # size of kernel in each conv layer
        conv_dilations=args.conv_dilations,  # dilation in each conv layer
        conv_bias=args.conv_bias,  # bias in each conv layer
        # number of filters in each recurrent layer
        recurrent_filters=args.recurrent_filters,
        # size of kernel in each recurrent layer
        recurrent_kernels=args.recurrent_kernels,
        recurrent_dilations=args.recurrent_dilations,  # dilation in each recurrent layer
        recurrent_bias=args.recurrent_bias,  # bias in each recurrent layer
        depth=args.depth,  # number of cascades
        conv_dim=args.conv_dim,  # dimensionality of input
        loss_fn=SSIMLoss() if args.loss_fn == "ssim" else L1Loss(),
        no_dc=args.no_dc,  # turn on/off DC component
        keep_eta=args.keep_eta,  # keep the eta signal
        use_sens_net=args.use_sens_net,  # use the sensitivity network
        sens_pools=args.sens_pools,  # number of pooling layers for sense est. U-Net
        sens_chans=args.sens_chans,  # number of top-level channels for sense est. U-Net
        sens_normalize=args.sens_normalize,  # normalize the sensitivity maps
        sens_mask_type=args.sens_mask_type,  # type of mask for sensitivity maps
        output_type=args.output_type,  # type of output
        fft_type=args.fft_type,  # type of FFT
    )

    return model.to(args.device, dtype=torch.float32)


def load_model(
    checkpoint_file: str,
) -> Tuple[Any, Union[torch.nn.DataParallel, torch.nn.Module], torch.optim.Optimizer]:
    """
    Load the model from disk.

    Args:
        checkpoint_file: File to load the model from.

    Returns:
        Tuple of (epoch, model, optimizer)
    """
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint["args"]
    model = build_model(args)

    if args.data_parallel:
        model = torch.nn.DataParallel(model)

    model.load_state_dict(checkpoint["model"])

    optimizer = build_optim(args, model.parameters())
    optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint, model, optimizer


def build_optim(args: argparse.Namespace, params: List[torch.nn.Parameter]) -> torch.optim.Optimizer:
    """
    Build the optimizer.

    Args:
        args: Arguments as parsed by argparse.
        params: Parameters to optimize.

    Returns:
        Optimizer to use for training.
    """
    if args.optimizer.upper() == "RMSPROP":
        optimizer = torch.optim.RMSprop(
            params, args.lr, weight_decay=args.weight_decay)
    if args.optimizer.upper() == "ADAM":
        optimizer = torch.optim.Adam(
            params, args.lr, weight_decay=args.weight_decay)
    if args.optimizer.upper() == "SGD":
        optimizer = torch.optim.SGD(
            params, args.lr, weight_decay=args.weight_decay)
    return optimizer


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
        for acc, cf in zip(args.accelerations, args.center_fractions):
            mask_func.append(create_mask_for_mask_type(
                args.mask_type, [cf] * 2, [acc] * 2))
    else:
        mask_func = create_mask_for_mask_type(
            args.mask_type, args.center_fractions, args.accelerations)

    train_transform = PhysicsInformedDataTransform(
        mask_func=mask_func,
        shift_mask=args.shift_mask,
        normalize_inputs=args.normalize_inputs,
        crop_size=args.crop_size,
        crop_before_masking=args.crop_before_masking,
        kspace_zero_filling_size=args.kspace_zero_filling_size,
        fft_type=args.fft_type,
        use_seed=True,
    )
    val_transform = PhysicsInformedDataTransform(
        mask_func=mask_func,
        shift_mask=args.shift_mask,
        normalize_inputs=args.normalize_inputs,
        crop_size=args.crop_size,
        crop_before_masking=args.crop_before_masking,
        kspace_zero_filling_size=args.kspace_zero_filling_size,
        fft_type=args.fft_type,
    )

    train_loader, val_loader, display_loader = create_training_loaders(
        args, train_transform, val_transform)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, args.lr_step_size, args.lr_gamma)

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
    parser = argparse.ArgumentParser(description="CIRIM")

    parser.add_argument("data_path", type=pathlib.Path,
                        help="Path to the data folder")
    parser.add_argument(
        "exp_dir", type=pathlib.Path, default="checkpoints", help="Path where model and results should be saved"
    )
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
    parser.add_argument("--num_cascades", type=int, default=1,
                        help="Number of cascades for the model")
    parser.add_argument("--time_steps", type=int,
                        default=8, help="Number of RIM steps")
    parser.add_argument("--recurrent_layer", type=str,
                        default="IndRNN", help="Recurrent layer type")
    parser.add_argument(
        "--conv_filters",
        nargs="+",
        default=[64, 64, 2],
        help="Number of filters the convolutional layers of for the model",
    )
    parser.add_argument(
        "--conv_kernels", nargs="+", default=[5, 3, 3], help="Kernel size for the convolutional layers of the model"
    )
    parser.add_argument(
        "--conv_dilations", nargs="+", default=[1, 2, 1], help="Dilations for the convolutional layers of the model"
    )
    parser.add_argument(
        "--conv_bias", nargs="+", default=[True, True, False], help="Bias for the convolutional layers of the model"
    )
    parser.add_argument(
        "--recurrent_filters",
        nargs="+",
        default=[64, 64, 0],
        help="Number of filters the recurrent layers of for the model",
    )
    parser.add_argument(
        "--recurrent_kernels", nargs="+", default=[1, 1, 0], help="Kernel size for the recurrent layers of the model"
    )
    parser.add_argument(
        "--recurrent_dilations", nargs="+", default=[1, 1, 0], help="Dilations for the recurrent layers of the model"
    )
    parser.add_argument(
        "--recurrent_bias", nargs="+", default=[True, True, False], help="Bias for the recurrent layers of the model"
    )
    parser.add_argument("--depth", type=int, default=2,
                        help="Depth of the model")
    parser.add_argument("--conv_dim", type=int, default=2,
                        help="Dimension of the convolutional layers")
    parser.add_argument("--loss_fn", type=str, default="l1",
                        help="Loss function to use")
    parser.add_argument("--no_dc", action="store_false",
                        default=True, help="Do not use DC component")
    parser.add_argument("--keep_eta", action="store_false",
                        default=True, help="Keep eta constant")
    parser.add_argument("--use_sens_net", action="store_true",
                        default=False, help="Use sensitivity net")
    parser.add_argument("--sens_pools", type=int, default=4,
                        help="Number of pools for the sensitivity net")
    parser.add_argument("--sens_chans", type=int, default=8,
                        help="Number of channels for the sensitivity net")
    parser.add_argument("--sens_normalize", action="store_true",
                        help="Normalize the sensitivity net")
    parser.add_argument(
        "--sens_mask_type", choices=["1D", "2D"], default="2D", help="Type of mask to use for the sensitivity net"
    )
    parser.add_argument(
        "--output_type", choices=["SENSE", "RSS"], default="SENSE", help="Type of output to use")
    parser.add_argument("--fft_type", type=str,
                        default="orthogonal", help="Type of FFT to use")
    parser.add_argument("--batch_size", default=1,
                        type=int, help="Mini batch size")
    parser.add_argument("--num_epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    parser.add_argument(
        "--optimizer", type=str, default="Adam", help="Optimizer to use choose between" "['Adam', 'SGD', 'RMSProp']"
    )
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--lr_step_size", type=int, default=40,
                        help="Period of learning rate decay")
    parser.add_argument("--lr_gamma", type=float, default=0.1,
                        help="Multiplicative factor of learning rate decay")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Strength of weight decay regularization")
    parser.add_argument("--report_interval", type=int,
                        default=150, help="Period of loss reporting")
    parser.add_argument(
        "--resume",
        action="store_true",
        help='If set, resume the training from a previous model checkpoint. "--checkpoint" should be set with this',
    )
    parser.add_argument("--checkpoint", type=str,
                        help='Path to an existing checkpoint. Used along with "--resume"')
    parser.add_argument("--exit_after_checkpoint", action="store_true",
                        help="If set, exit after loading a checkpoint")
    parser.add_argument("--seed", default=42, type=int,
                        help="Seed for random number generators")
    parser.add_argument(
        "--data_parallel", action="store_true", help="If set, use multiple GPUs using data parallelism"
    )
    parser.add_argument("--device", type=str, default="cuda",
                        help="Which device to run on")

    return parser


if __name__ == "__main__":
    ARGS = create_arg_parser().parse_args()
    random.seed(ARGS.seed)
    np.random.seed(ARGS.seed)
    torch.manual_seed(ARGS.seed)
    main(ARGS)
