# coding=utf-8
from pathlib import Path

import h5py
import numpy as np
from runstats import Statistics

from mridc.collections.common.parts import center_crop, utils
from mridc.collections.reconstruction.metrics.reconstruction_metrics import mse, nmse, psnr, ssim

METRIC_FUNCS = {"MSE": mse, "NMSE": nmse, "PSNR": psnr, "SSIM": ssim}


class Metrics:
    """Maintains running statistics for a given collection of metrics."""

    def __init__(self, metric_funcs):
        """
        Args:
            metric_funcs (dict): A dict where the keys are metric names and the
                values are Python functions for evaluating that metric.
        """
        self.metrics_scores = {metric: Statistics() for metric in metric_funcs}

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

        return res


def main(args):
    # if json file
    if args.targets_dir.endswith(".json"):
        import json

        with open(args.targets_dir, "r") as f:
            targets = json.load(f)
        targets = [Path(target) for target in targets]
    else:
        targets = list(Path(args.targets_dir).iterdir())

    scores = Metrics(METRIC_FUNCS)

    for target in targets:
        reconstruction = h5py.File(Path(args.reconstructions_dir) / str(target).split("/")[-1], "r")["reconstruction"][
            ()
        ].squeeze()
        if "reconstruction_sense" in h5py.File(target, "r").keys():
            target = h5py.File(target, "r")["reconstruction_sense"][()].squeeze()
        elif "reconstruction_rss" in h5py.File(target, "r").keys():
            target = h5py.File(target, "r")["reconstruction_rss"][()].squeeze()
        elif "reconstruction" in h5py.File(target, "r").keys():
            target = h5py.File(target, "r")["reconstruction"][()].squeeze()
        else:
            target = h5py.File(target, "r")["target"][()].squeeze()

        crop_size = [320, 320]
        crop_size[0] = target.shape[-2] if target.shape[-2] < int(crop_size[0]) else int(crop_size[0])
        crop_size[1] = target.shape[-1] if target.shape[-1] < int(crop_size[1]) else int(crop_size[1])
        crop_size[0] = reconstruction.shape[-2] if reconstruction.shape[-2] < int(crop_size[0]) else int(crop_size[0])
        crop_size[1] = reconstruction.shape[-1] if reconstruction.shape[-1] < int(crop_size[1]) else int(crop_size[1])

        target = center_crop(target, crop_size)
        reconstruction = center_crop(reconstruction, crop_size)

        for sl in range(target.shape[0]):
            # for sl in range(11):
            target[sl] = target[sl] / np.max(np.abs(target[sl]))
            reconstruction[sl] = reconstruction[sl] / np.max(np.abs(reconstruction[sl]))

        target = np.abs(target)
        reconstruction = np.abs(reconstruction)

        target = target / np.max(np.abs(target))
        reconstruction = reconstruction / np.max(np.abs(reconstruction))

        scores.push(target, reconstruction)

    print(scores.__repr__()[:-1])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("targets_dir", type=str)
    parser.add_argument("reconstructions_dir", type=str)
    args = parser.parse_args()
    main(args)
