# coding=utf-8
__author__ = "Dimitrios Karkalousos"

import argparse
from argparse import ArgumentParser
from pathlib import Path

import h5py
import numpy as np
from SimpleITK import (
    GetImageFromArray,
    ReadImage,
    WriteImage,
)
from tqdm import tqdm


def main(input_path, output_path, stored_dicom_path):
    """
    Export a reconstruction to DICOM format.

    Args:
        input_path (): Path to the reconstruction.
        output_path ():  Path to the output directory.
        stored_dicom_path (): Path to the directory containing the DICOM files.
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

        # Create the SimpleITK image
        data = GetImageFromArray(data)

        if stored_dicom_path is not None:
            image = ReadImage(stored_dicom_path)
            origin = image.GetOrigin()
            spacing = image.GetSpacing()
            direction = image.GetDirection()
            data.SetOrigin((origin[2], origin[0], origin[1]))
            data.SetSpacing((spacing[2], spacing[0], spacing[1]))
            data.SetDirection(
                (
                    direction[3],
                    direction[4],
                    direction[5],  # y
                    direction[6],
                    direction[7],
                    direction[8],  # z
                    direction[0],
                    direction[1],
                    direction[2],  # x
                )
            )

        # Write the image to the output directory
        WriteImage(data, str(output_path / file.name) + ".dcm")


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input_path", type=Path, help="Path to the reconstructions saved as h5 files.")
    parser.add_argument("output_path", type=Path, help="Path to export the reconstructions as dicom files.")
    parser.add_argument("--stored_dicom_path", type=str, help="Path to scanner's stored dicom files.")
    args = parser.parse_args()

    args.output_path.mkdir(exist_ok=True, parents=True)

    main(args.input_path, args.output_path, args.stored_dicom_path)
