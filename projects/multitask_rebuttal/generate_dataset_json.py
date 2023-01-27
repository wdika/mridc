# coding=utf-8
__author__ = "Dimitrios Karkalousos"

import argparse
import json
import random
from pathlib import Path

import numpy as np


def read_h5_files(dataset_path):
    """Read all h5 files in a directory"""
    return list(Path(dataset_path).iterdir())


def generate_fold(filenames):
    """Generate a train, val and test set from a list of filenames"""

    # Path to str
    filenames = [str(filename) for filename in filenames]

    # shuffle the filenames
    random.shuffle(filenames)

    # split the filenames into train, val and test with 70%, 15% and 15% respectively
    train_fnames = np.array(filenames[: int(len(filenames) * 0.7)])
    # remove train filenames from all filenames
    filenames = np.setdiff1d(filenames, train_fnames)
    val_fnames = np.array(filenames[: int(len(filenames) * 0.5)])
    # remove val filenames from all filenames
    filenames = np.setdiff1d(filenames, val_fnames)
    test_fnames = np.array(filenames)

    return train_fnames.tolist(), val_fnames.tolist(), test_fnames.tolist()


def main(args):
    if args.data_path is not None:
        # read all h5 files in the data directory
        all_filenames = read_h5_files(args.data_path)
    else:
        # read all h5 files in the train, val and test directories
        train_filenames = read_h5_files(args.train_path)
        val_filenames = read_h5_files(args.val_path)
        test_filenames = read_h5_files(args.test_path)
        # merge the train, val and test filenames
        all_filenames = train_filenames + val_filenames + test_filenames

    # create n folds
    folds = [generate_fold(all_filenames) for _ in range(args.nfolds)]

    # create a directory to store the folds
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # write each fold to a json file
    for i, fold in enumerate(folds):
        train_set, val_set, test_set = fold

        # write the train, val and test filenames to a json file
        with open(output_path / f"fold_{i}_train.json", "w") as f:
            json.dump(train_set, f)
        with open(output_path / f"fold_{i}_val.json", "w") as f:
            json.dump(val_set, f)
        with open(output_path / f"fold_{i}_test.json", "w") as f:
            json.dump(test_set, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=Path,
        default=None,
        help="Path to the data directory, if the data is not already split into train, val and test. "
        "Then the --train_path, --val_path and --test_path arguments are ignored.",
    )
    parser.add_argument(
        "--train_path",
        type=Path,
        default=None,
        help="Path to the train directory, if data are already split into train, val and test. "
        "If --data_path is provided, then this argument is ignored.",
    )
    parser.add_argument(
        "--val_path",
        type=Path,
        default=None,
        help="Path to the val directory, if data are already split into train, val and test. "
        "If --data_path is provided, then this argument is ignored.",
    )
    parser.add_argument(
        "--test_path",
        type=Path,
        default=None,
        help="Path to the test directory, if data are already split into train, val and test. "
        "If --data_path is provided, then this argument is ignored.",
    )
    parser.add_argument("--output_path", type=Path, default="data/folds", help="Path to the output directory.")
    parser.add_argument("--nfolds", type=int, default=5, help="Number of folds to create.")
    args = parser.parse_args()
    main(args)
