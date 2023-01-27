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
from mridc.collections.segmentation import metrics


def process_segmentation_labels(
    segmentation_labels: np.ndarray,
    segmentation_classes: int,
    segmentation_classes_to_remove: Tuple[int, ...],
    segmentation_classes_to_combine: Tuple[int, ...],
    segmentation_classes_to_separate: Tuple[int, ...],
    segmentation_classes_thresholds: Tuple[float, ...] = None,
    consecutive_slices: int = 1,
) -> np.ndarray:
    """
    Process segmentation labels to remove, combine, and separate classes.

    Parameters
    ----------
    segmentation_labels : np.ndarray
        The segmentation labels. The shape should be (num_slices, height, width) or (height, width).
    segmentation_classes : int
        The number of segmentation classes.
    segmentation_classes_to_remove : Tuple[int, ...]
        The segmentation classes to remove.
    segmentation_classes_to_combine : Tuple[Tuple[int, ...], ...]
        The segmentation classes to combine.
    segmentation_classes_to_separate : Tuple[Tuple[int, ...], ...]
        The segmentation classes to separate.
    segmentation_classes_thresholds : Tuple[float, ...]
        The segmentation classes thresholds.
    consecutive_slices : int
        The number of consecutive slices to consider.

    Returns
    -------
    np.ndarray
        The processed segmentation labels.
    """
    # make sure that the segmentation dim will be the last one
    segmentation_labels_dim = -1
    for dim in range(segmentation_labels.ndim):
        if segmentation_labels.shape[dim] == segmentation_classes:
            segmentation_labels_dim = dim

    if segmentation_labels.ndim == 2:
        segmentation_labels = np.expand_dims(segmentation_labels, axis=0)

    if segmentation_labels.ndim == 3 and segmentation_labels_dim == 0:
        segmentation_labels = np.transpose(segmentation_labels, (1, 2, 0))
    elif segmentation_labels.ndim == 3 and segmentation_labels_dim == 1:
        segmentation_labels = np.transpose(segmentation_labels, (0, 2, 1))
    elif segmentation_labels.ndim == 4 and segmentation_labels_dim == 0:
        segmentation_labels = np.transpose(segmentation_labels, (1, 2, 3, 0))
    elif segmentation_labels.ndim == 4 and segmentation_labels_dim == 1:
        segmentation_labels = np.transpose(segmentation_labels, (0, 2, 3, 1))
    elif segmentation_labels.ndim == 4 and segmentation_labels_dim == 2:
        segmentation_labels = np.transpose(segmentation_labels, (0, 1, 3, 2))

    removed_classes = 0

    # check if we need to remove any classes, e.g. background
    if not is_none(segmentation_classes_to_remove):
        segmentation_labels = np.stack(
            [
                segmentation_labels[..., x]
                for x in range(segmentation_labels.shape[-1])
                if x not in segmentation_classes_to_remove  # type: ignore
            ],
            axis=-1,
        )
        removed_classes += len(segmentation_classes_to_remove)  # type: ignore

    # check if we need to combine any classes, e.g. White Matter and Gray Matter
    if not is_none(segmentation_classes_to_combine):
        segmentation_labels_to_combine = []
        segmentation_labels_to_keep = []
        for x in range(segmentation_labels.shape[-1]):
            if x in segmentation_classes_to_combine:  # type: ignore
                segmentation_labels_to_combine.append(segmentation_labels[..., x - removed_classes])
            else:
                segmentation_labels_to_keep.append(segmentation_labels[..., x - removed_classes])
        segmentation_labels_to_combine = np.expand_dims(
            np.sum(np.stack(segmentation_labels_to_combine, axis=-1), axis=-1), axis=-1
        )
        segmentation_labels_to_keep = np.stack(segmentation_labels_to_keep, axis=-1)

        if segmentation_classes_to_remove is not None and (
            0 in segmentation_classes_to_remove or "0" in segmentation_classes_to_remove
        ):
            # if background is removed, we can stack the combined labels with the rest straight away
            segmentation_labels = np.concatenate(
                [segmentation_labels_to_combine, segmentation_labels_to_keep], axis=-1
            )
        else:
            # if background is not removed, we need to add it back as new background channel
            segmentation_labels = np.concatenate(
                [
                    np.expand_dims(segmentation_labels[..., 0], axis=-1),
                    segmentation_labels_to_combine,
                    segmentation_labels_to_keep,
                ],
                axis=-1,
            )

        removed_classes += len(segmentation_classes_to_combine) - 1  # type: ignore

    # check if we need to separate any classes, e.g. pathologies from White Matter and Gray Matter
    if not is_none(segmentation_classes_to_separate):
        for x in segmentation_classes_to_separate:  # type: ignore
            segmentation_class_to_separate = segmentation_labels[..., x - removed_classes]
            for i in range(segmentation_labels.shape[-1]):
                if i == x - removed_classes:
                    continue
                segmentation_labels[..., i][segmentation_class_to_separate > 0] = 0

    segmentation_labels = (
        np.moveaxis(segmentation_labels, -1, 0)
        if segmentation_labels.ndim == 3
        else np.moveaxis(segmentation_labels, -1, 1)
    )

    # # threshold probability maps if any threshold is given
    if not is_none(segmentation_classes_thresholds):
        for i, voxel_thres in enumerate(segmentation_classes_thresholds):  # type: ignore
            if not is_none(voxel_thres):
                segmentation_labels[..., i] = segmentation_labels[..., i] > float(voxel_thres)

    if consecutive_slices == 1:
        if segmentation_labels.shape[1] == segmentation_classes:
            segmentation_labels = np.moveaxis(segmentation_labels, 1, 0)
        elif segmentation_labels.shape[2] == segmentation_classes:
            segmentation_labels = np.moveaxis(segmentation_labels, 2, 0)

    return segmentation_labels


def main(args):
    df = pd.DataFrame(columns=["Method", "Fold", "Subject", "Dice", "Dice_Lesions"])
    # Load ground truth and predicted labels
    if args.targets_path_json.suffix == ".json":
        targets = [Path(x) for x in json.load(open(args.targets_path_json, "r"))]
    else:
        targets = list(Path(args.targets_path_json).iterdir())
    predictions = list(Path(args.predictions_path).iterdir())
    # Evaluate performance
    for target, prediction in tqdm(zip(targets, predictions)):
        fname = prediction.name
        target = process_segmentation_labels(
            h5py.File(target, "r")["segmentation"][()],
            segmentation_classes=2,
            segmentation_classes_to_combine=[1, 2],
            segmentation_classes_to_remove=[0],
            segmentation_classes_to_separate=[3],
            segmentation_classes_thresholds=[0.5, 0.5],
            consecutive_slices=1,
        )
        target = torch.from_numpy(np.moveaxis(target, 0, 1))

        prediction = torch.from_numpy(h5py.File(prediction, "r")["segmentation"][()].squeeze())

        # normalize with max per slice
        for i in range(target.shape[0]):
            target[i] = target[i] / torch.max(torch.abs(target[i]))
        target = np.abs(target)

        # normalize with max per slice
        for i in range(prediction.shape[0]):
            prediction[i] = prediction[i] / torch.max(torch.abs(prediction[i]))
        prediction = torch.abs(prediction)

        # ignore pandas deprecation warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            df = df.append(
                {
                    "Method": args.method,
                    "Fold": args.fold,
                    "Subject": fname,
                    "Dice": metrics.dice_metric(prediction, target, include_background=True),
                    "Dice_Lesions": metrics.dice_metric(prediction, target, include_background=False),
                },
                ignore_index=True,
            )

    # save to csv
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
