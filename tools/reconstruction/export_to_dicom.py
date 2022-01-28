# coding=utf-8
__author__ = "Dimitrios Karkalousos"

import argparse
from argparse import ArgumentParser
from pathlib import Path

import h5py
import numpy as np
from SimpleITK import GetImageFromArray, WriteImage
from tqdm import tqdm


def main(input_path, output_path):
    """
    Export a reconstruction to DICOM format.

    Args:
        input_path (): Path to the reconstruction.
        output_path ():  Path to the output directory.
    """
    # Read files.
    files = list(Path(input_path).iterdir())

    for file in tqdm(files):
        # Load the reconstruction for each file.
        data = h5py.File(file, "r")["reconstruction"][()]

        # Get the magnitude and remove the extra dimension.
        data = np.abs(data).squeeze(1)

        # Remove nan and inf values, and normalize to [0, 1].
        data = np.where(np.isnan(data), 0, data)
        data = np.where(np.isinf(data), 1, data)
        data = data / np.max(data)

        # Normalize the data to be converted as uint16, necessary for DICOM conversion.
        factor = np.max(data) / 65535
        data = (data / factor).astype(np.uint16)

        # Rotate the data
        data = np.flip(data, axis=(0, 1, 2))

        # Flip left-right the data to match the orientation of the scanner
        data = data[:, :, ::-1]

        # Create the SimpleITK image
        data = GetImageFromArray(data)

        # Write the image to the output directory
        WriteImage(data, str(output_path / file.name) + ".dcm")


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input_path", type=Path, help="Path to the reconstructions saved as h5 files.")
    parser.add_argument("output_path", type=Path, help="Path to export the reconstructions as dicom files.")
    args = parser.parse_args()

    args.output_path.mkdir(exist_ok=True, parents=True)

    main(args.input_path, args.output_path)
