# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

from argparse import ArgumentParser
from pathlib import Path

import h5py
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


def save_png(data_dir: Path, out_dir: Path) -> None:
    """
    Saves the reconstructions to png files.

    Parameters
    ----------
    data_dir: Path
    out_dir: Path

    Returns
    -------
    None
    """
    for fname in tqdm(list(data_dir.iterdir())):
        with h5py.File(fname, "r") as hf:
            if "reconstruction" in hf:
                image = hf["reconstruction"][()]
            elif "reconstruction_rss" in hf:
                image = hf["reconstruction_rss"][()]
            elif "reconstruction_sense" in hf:
                image = hf["reconstruction_sense"][()]
            elif "pics" in hf:
                image = hf["pics"][()]
            elif "target" in hf:
                image = hf["target"][()]

            image = image / np.max(np.abs(image))

            name = str(fname.name).split(".")[0]
            out_path = out_dir / name
            out_path.mkdir(exist_ok=True, parents=True)

            for sl in range(image.shape[0]):
                im = image[sl].squeeze()
                plt.figure(figsize=(15, 15))
                plt.subplot(2, 2, 1)
                plt.imshow(np.abs(im), cmap="gray")
                plt.title("Magnitude")
                plt.axis("off")
                plt.subplot(2, 2, 2)
                plt.imshow(np.angle(im), cmap="gray")
                plt.title("Phase")
                plt.axis("off")
                plt.subplot(2, 2, 3)
                plt.imshow(np.real(im), cmap="gray")
                plt.title("Real")
                plt.axis("off")
                plt.subplot(2, 2, 4)
                plt.imshow(np.imag(im), cmap="gray")
                plt.title("Imaginary")
                plt.axis("off")
                plt.savefig(f'{str(out_path)}/{str(sl)}.png')
                plt.close()


def create_arg_parser():
    """
    Creates an argument parser for the script.

    Returns:
        argparse.ArgumentParser: The argument parser.
    """
    parser = ArgumentParser()

    parser.add_argument("data_path", type=Path, help="Path to the data")
    parser.add_argument("output_path", type=Path, help="Path to save the reconstructions to")

    return parser


if __name__ == "__main__":
    args = create_arg_parser().parse_args()
    save_png(args.data_path, args.output_path)
