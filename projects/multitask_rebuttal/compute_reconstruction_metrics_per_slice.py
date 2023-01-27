# coding=utf-8
import argparse
import json
import warnings
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from mridc.collections.common.parts import is_none
from mridc.collections.reconstruction import metrics


def main(args):
    df = pd.DataFrame(columns=["Method", "Fold", "Subject", "SSIM", "PSNR"])
    # Load ground truth and predicted labels
    if args.targets_path_json.suffix == ".json":
        targets = [Path(x) for x in json.load(open(args.targets_path_json, "r"))]
    else:
        targets = list(Path(args.targets_path_json).iterdir())
    predictions = list(Path(args.predictions_path).iterdir())
    # Evaluate performance
    for target, prediction in tqdm(zip(targets, predictions)):
        fname = prediction.name
        target = np.expand_dims(h5py.File(target, "r")["reconstruction_sense"][()], 1)
        for i in range(target.shape[0]):
            target[i] = target[i] / np.max(np.abs(target[i]))
        target = np.abs(target)

        # normalize with max per slice
        prediction = h5py.File(prediction, "r")["reconstruction"][()]
        for i in range(prediction.shape[0]):
            prediction[i] = prediction[i] / np.max(np.abs(prediction[i]))
        prediction = np.abs(prediction)

        for i in range(prediction.shape[0]):
            # ignore pandas deprecation warning
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                df = df.append(
                    {
                        "Method": args.method,
                        "Fold": args.fold,
                        "Subject": fname,
                        "Slice": i,
                        "SSIM": metrics.ssim(target[i], prediction[i], maxval=target[i].max() - target[i].min()),
                        "PSNR": metrics.psnr(target[i], prediction[i], maxval=target[i].max() - target[i].min()),
                    },
                    ignore_index=True,
                )

    # # save to csv
    parent_output_path = Path(args.output_path).parent
    parent_output_path.mkdir(parents=True, exist_ok=True)

    # save to csv, if csv exists append to it
    if Path(args.output_path).exists():
        df.to_csv(args.output_path, mode="a", header=False, index=False)
    else:
        df.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("targets_path_json", type=Path)
    parser.add_argument("predictions_path", type=Path)
    parser.add_argument("output_path", type=str)
    parser.add_argument("method", type=str)
    parser.add_argument("fold", type=int)
    args = parser.parse_args()
    main(args)
