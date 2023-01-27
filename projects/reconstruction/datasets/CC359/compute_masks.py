# coding=utf-8
__author__ = "Dimitrios Karkalousos"

import argparse
import pathlib

import h5py
import numpy as np
from tqdm import tqdm


def main(args):
    data_dir = args.data_dir
    masks_dir = args.masks_dir

    # Create output directory
    output_dir = args.output_dir
    output_dir.mkdir(exist_ok=True)
    output_dir = output_dir / str(data_dir).split("/")[-1]
    output_dir.mkdir(exist_ok=True)

    for data_file in tqdm(data_dir.glob("*.h5")):
        with h5py.File(data_file, "r") as f:
            # load k-space data to get the shape
            data = f["kspace"]

            # load respective 5x mask
            mask_5x = np.load(masks_dir / f"R5_{data.shape[1]}x{data.shape[2]}.npy")
            # concatenate the mask to match the number of slices, original there are 100 slices/masks
            mask_5x = np.concatenate((mask_5x, mask_5x), axis=0)
            random_slices = np.random.choice(mask_5x.shape[0], (data.shape[0] - mask_5x.shape[0]), replace=False)
            mask_5x = np.concatenate((mask_5x, mask_5x[random_slices]), axis=0)

            # load respective 10x mask
            mask_10x = np.load(masks_dir / f"R10_{data.shape[1]}x{data.shape[2]}.npy")
            # concatenate the mask to match the number of slices, original there are 100 slices/masks
            mask_10x = np.concatenate((mask_10x, mask_10x), axis=0)
            random_slices = np.random.choice(
                mask_10x.shape[0],
                int(mask_10x.shape[0] * (data.shape[0] - mask_10x.shape[0]) / mask_10x.shape[0]),
                replace=False,
            )
            mask_10x = np.concatenate((mask_10x, mask_10x[random_slices]), axis=0)

        with h5py.File(output_dir / data_file.name, "w") as f:
            f.create_dataset("mask_5x", data=mask_5x)
            f.create_dataset("mask_10x", data=mask_10x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=pathlib.Path, help="Path to the data directory.")
    parser.add_argument("masks_dir", type=pathlib.Path, help="Path to the masks directory.")
    parser.add_argument("output_dir", type=pathlib.Path, help="Path to the output directory.")
    main(parser.parse_args())
