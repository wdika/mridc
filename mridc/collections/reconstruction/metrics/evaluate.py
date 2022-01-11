# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Parts of the code have been taken from https://github.com/facebookresearch/fastMRI

import argparse
import pathlib
from argparse import ArgumentParser
from os.path import exists

import h5py
import numpy as np
import pandas as pd
import torch
from runstats import Statistics
from skimage.filters import threshold_otsu
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.morphology import convex_hull_image
from tqdm import tqdm

from mridc.collections.common.parts.fft import ifft2c
from mridc.collections.common.parts.utils import complex_conj, complex_mul, tensor_to_complex_np, to_tensor
from mridc.collections.reconstruction.parts.utils import center_crop


def mse(gt: np.ndarray, pred: np.ndarray) -> float:
    """Compute Mean Squared Error (MSE)"""
    return np.mean((gt - pred) ** 2)  # type: ignore


def nmse(gt: np.ndarray, pred: np.ndarray) -> float:
    """Compute Normalized Mean Squared Error (NMSE)"""
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2


def psnr(gt: np.ndarray, pred: np.ndarray, maxval: np.ndarray = None) -> float:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    if maxval is None:
        maxval = np.max(gt)
    return peak_signal_noise_ratio(gt, pred, data_range=maxval)


def ssim(gt: np.ndarray, pred: np.ndarray, maxval: np.ndarray = None) -> float:
    """Compute Structural Similarity Index Metric (SSIM)"""
    if not gt.ndim == 3:
        raise ValueError("Unexpected number of dimensions in ground truth.")
    if not gt.ndim == pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")

    maxval = np.max(gt) if maxval is None else maxval

    _ssim = 0
    for slice_num in range(gt.shape[0]):
        _ssim = _ssim + structural_similarity(gt[slice_num], pred[slice_num], data_range=maxval)

    return _ssim / gt.shape[0]


METRIC_FUNCS = dict(MSE=mse, NMSE=nmse, PSNR=psnr, SSIM=ssim)


class Metrics:
    """Maintains running statistics for a given collection of metrics."""

    def __init__(self, metric_funcs, output_path, method):
        """
        Args:
            metric_funcs (dict): A dict where the keys are metric names and the
                values are Python functions for evaluating that metric.
            output_path:
            method:
        """
        self.metrics_scores = {metric: Statistics() for metric in metric_funcs}
        self.output_path = output_path
        self.method = method

    def push(self, target, recons):
        """
        Pushes a new batch of metrics to the running statistics.

        Args:
            target: target image
            recons: reconstructed image

        Returns:
            dict: A dict where the keys are metric names and the values are
        """
        for metric, func in METRIC_FUNCS.items():
            self.metrics_scores[metric].push(func(target, recons))

    def means(self):
        """
        Mean of the means of each metric.

        Returns:
            dict: A dict where the keys are metric names and the values are
        """
        return {metric: stat.mean() for metric, stat in self.metrics_scores.items()}

    def stddevs(self):
        """
        Standard deviation of the means of each metric.

        Returns:
            dict: A dict where the keys are metric names and the values are
        """
        return {metric: stat.stddev() for metric, stat in self.metrics_scores.items()}

    def __repr__(self):
        """
        Representation of the metrics.

        Returns:
            str: A string representation of the metrics.
        """
        means = self.means()
        stddevs = self.stddevs()
        metric_names = sorted(list(means))

        res = " ".join(f"{name} = {means[name]:.4g} +/- {2 * stddevs[name]:.4g}" for name in metric_names) + "\n"

        with open(self.output_path + "metrics.txt", "a") as output:
            output.write(f"{self.method}: {res}")

        return res


def evaluate(
    arguments, reconstruction_key, mask_background, output_path, method, acc, no_params, slice_start, slice_end
):
    """
    Evaluate the reconstruction.
    TODO: Refine this function.
    """
    _metrics = Metrics(METRIC_FUNCS, output_path, method) if arguments.type == "mean_std" else {}

    for tgt_file in tqdm(arguments.target_path.iterdir()):
        if exists(arguments.predictions_path / tgt_file.name):
            with h5py.File(tgt_file, "r") as target, h5py.File(
                arguments.predictions_path / tgt_file.name, "r"
            ) as recons:

                if arguments.sense_path is not None:
                    sense = to_tensor(h5py.File(arguments.sense_path / tgt_file.name, "r")["sensitivity_map"][()])
                elif "sensitivity_map" in target:
                    sense = to_tensor(target["sensitivity_map"][()])

                target = np.abs(
                    tensor_to_complex_np(
                        torch.sum(
                            complex_mul(
                                ifft2c(
                                    to_tensor(target["kspace"][()]),
                                    fft_type="data" if "data" in str(arguments.sense_path).lower() else "nofastmri",
                                ),
                                complex_conj(sense),
                            ),
                            1,
                        )
                    )
                )

                recons = recons[reconstruction_key][()]

                if recons.ndim == 4:
                    recons = recons.squeeze(1)

                if arguments.crop_size is not None:
                    crop_size = arguments.crop_size
                    crop_size[0] = target.shape[-2] if target.shape[-2] < int(crop_size[0]) else int(crop_size[0])
                    crop_size[1] = target.shape[-1] if target.shape[-1] < int(crop_size[1]) else int(crop_size[1])
                    crop_size[0] = recons.shape[-2] if recons.shape[-2] < int(crop_size[0]) else int(crop_size[0])
                    crop_size[1] = recons.shape[-1] if recons.shape[-1] < int(crop_size[1]) else int(crop_size[1])

                    target = center_crop(target, crop_size)
                    recons = center_crop(recons, crop_size)

                if mask_background:
                    for sl in range(target.shape[0]):
                        mask = convex_hull_image(
                            np.where(np.abs(target[sl]) > threshold_otsu(np.abs(target[sl])), 1, 0)  # type: ignore
                        )
                        target[sl] = target[sl] * mask
                        recons[sl] = recons[sl] * mask

                if slice_start is not None:
                    target = target[slice_start:]
                    recons = recons[slice_start:]

                if slice_end is not None:
                    target = target[:slice_end]
                    recons = recons[:slice_end]

                for sl in range(target.shape[0]):
                    target[sl] = target[sl] / np.max(np.abs(target[sl]))
                    recons[sl] = recons[sl] / np.max(np.abs(recons[sl]))

                target = np.abs(target)
                recons = np.abs(recons)

                if arguments.type == "mean_std":
                    _metrics.push(target, recons)
                else:
                    _target = np.expand_dims(target, 1)
                    _recons = np.expand_dims(recons, 1)
                    for sl in range(target.shape[0]):
                        _metrics["FNAME"] = tgt_file.name
                        _metrics["SLICE"] = sl
                        _metrics["ACC"] = acc
                        _metrics["METHOD"] = method
                        _metrics["MSE"] = [mse(target[sl], recons[sl])]
                        _metrics["NMSE"] = [nmse(target[sl], recons[sl])]
                        _metrics["PSNR"] = [psnr(target[sl], recons[sl])]
                        _metrics["SSIM"] = [ssim(_target[sl], _recons[sl])]
                        _metrics["PARAMS"] = no_params

                        if not exists(arguments.output_path):
                            pd.DataFrame(columns=_metrics.keys()).to_csv(arguments.output_path, index=False, mode="w")
                        pd.DataFrame(_metrics).to_csv(arguments.output_path, index=False, header=False, mode="a")

    return _metrics


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("target_path", type=pathlib.Path, help="Path to the ground truth data")
    parser.add_argument("predictions_path", type=pathlib.Path, help="Path to reconstructions")
    parser.add_argument("output_path", type=str, help="Path to save the metrics")
    parser.add_argument("--sense_path", type=pathlib.Path, help="Path to the sense data")
    parser.add_argument(
        "--challenge",
        choices=["singlecoil", "multicoil", "multicoil_sense"],
        default="multicoil_sense",
        help="Which challenge",
    )
    parser.add_argument("--crop_size", nargs="+", default=None, help="Set crop size.")
    parser.add_argument("--method", type=str, required=True, help="Model's name to evaluate")
    parser.add_argument("--acceleration", type=int, required=True, default=None)
    parser.add_argument("--no_params", type=str, required=True, default=None)
    parser.add_argument(
        "--acquisition",
        choices=["CORPD_FBK", "CORPDFS_FBK", "AXT1", "AXT1PRE", "AXT1POST", "AXT2", "AXFLAIR"],
        default=None,
        help="If set, only volumes of the specified acquisition type are used "
        "for evaluation. By default, all volumes are included.",
    )
    parser.add_argument("--mask_background", action="store_true", help="Toggle to mask background")
    parser.add_argument(
        "--type", choices=["mean_std", "all_slices"], default="mean_std", help="Toggle to mask background"
    )
    parser.add_argument("--slice_start", type=int, help="Select to skip first slices")
    parser.add_argument("--slice_end", type=int, help="Select to skip last slices")

    args = parser.parse_args()

    if args.challenge == "multicoil":
        recons_key = "reconstruction_rss"
    elif args.challenge == "multicoil_sense":
        recons_key = "reconstruction_sense"
    elif args.challenge == "singlecoil":
        recons_key = "reconstruction_esc"
    else:
        recons_key = "reconstruction"

    metrics = evaluate(
        args,
        recons_key,
        args.mask_background,
        args.output_path,
        args.method,
        args.acceleration,
        args.no_params,
        args.slice_start,
        args.slice_end,
    )

    if args.type == "mean_std":
        print(metrics)
    elif args.type == "all_slices":
        print("Done, csv file saved!")
