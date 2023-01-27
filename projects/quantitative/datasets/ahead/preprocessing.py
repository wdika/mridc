# coding=utf-8
__author__ = "Dimitrios Karkalousos"

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import h5py
import ismrmrd
import numpy as np
import torch
from scipy.ndimage import binary_dilation
from skimage.filters import threshold_otsu
from skimage.morphology import convex_hull_image
from tqdm import tqdm

from mridc.collections.quantitative.parts.transforms import R2star_B0_S0_phi_mapping


def __preprocess_ahead_raw_data__(raw_data_file: str) -> np.ndarray:
    """
    Preprocess the raw data of the AHEAD dataset.

    Parameters
    ----------
    raw_data_file : str
        Path to the raw data and coil sensitivities of the AHEAD dataset.

    Returns
    -------
    kspace: np.ndarray
        The k-space data.
    """
    dataset = ismrmrd.Dataset(raw_data_file, "dataset", create_if_needed=False)
    number_of_acquisitions = dataset.number_of_acquisitions()

    # find the first no noise scan
    first_scan = 0
    for i in tqdm(range(number_of_acquisitions)):
        head = dataset.read_acquisition(i).getHead()
        if head.isFlagSet(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):
            first_scan = i
            break

    meas = []
    for i in tqdm(range(first_scan, number_of_acquisitions)):
        acq = dataset.read_acquisition(i)
        meas.append(acq)

    hdr = ismrmrd.xsd.CreateFromDocument(dataset.read_xml_header())

    # Matrix size
    enc = hdr.encoding[0]
    enc_Nx = enc.encodedSpace.matrixSize.x
    enc_Ny = enc.encodedSpace.matrixSize.y
    enc_Nz = enc.encodedSpace.matrixSize.z

    nCoils = hdr.acquisitionSystemInformation.receiverChannels

    nslices = enc.encodingLimits.slice.maximum + 1 if enc.encodingLimits.slice is not None else 1
    nreps = enc.encodingLimits.repetition.maximum + 1 if enc.encodingLimits.repetition is not None else 1
    ncontrasts = enc.encodingLimits.contrast.maximum + 1 if enc.encodingLimits.contrast is not None else 1

    # initialize k-space array
    Kread = np.zeros((enc_Nx, enc_Ny, enc_Nz, nCoils), dtype=np.complex64)

    # Select the appropriate measurements from the data
    for acq in tqdm(meas):
        head = acq.getHead()
        if head.idx.contrast == ncontrasts - 1 and head.idx.repetition == nreps - 1 and head.idx.slice == nslices - 1:
            head = acq.getHead()
            ky = head.idx.kspace_encode_step_1
            kz = head.idx.kspace_encode_step_2
            Kread[:, ky, kz, :] = np.transpose(acq.data, (1, 0))

    return Kread


def __preprocess_ahead_coil_sensitivities__(coil_sensitivities_file: str) -> np.ndarray:
    """
    Preprocess the coil sensitivities of the AHEAD dataset.

    Parameters
    ----------
    coil_sensitivities_file : str
        Path to the coil sensitivities of the AHEAD dataset.

    Returns
    -------
    coil_sensitivities: np.ndarray
        The coil sensitivities.
    """
    # load the coil sensitivities
    coil_sensitivities = h5py.File(coil_sensitivities_file, "r")

    # get the coil sensitivities
    coil_sensitivities_real = np.array(coil_sensitivities["0real"])
    coil_sensitivities_imag = np.array(coil_sensitivities["1imag"])
    coil_sensitivities = coil_sensitivities_real + 1j * coil_sensitivities_imag

    # transpose to get the correct shape, i.e. (x, y, z, coils)
    coil_sensitivities = np.transpose(coil_sensitivities, (3, 2, 1, 0))

    return coil_sensitivities


def __get_plane__(data: np.ndarray, data_on_kspace: bool = True, plane: str = "sagittal") -> np.ndarray:
    """
    Get the given plane from the data.

    Parameters
    ----------
    data : np.ndarray
        The data to get the plane from.
    data_on_kspace : bool, optional
        Whether the data is on the kspace or not. The default is True.
    plane : str, optional
        The plane to get the kspace and coil sensitivities from. The default is "sagittal".

    Returns
    -------
    data: np.ndarray
        The data of the given plane.
    """
    if not data_on_kspace:
        data = np.fft.fftn(data, axes=(0, 1, 2))

    if plane == "axial":
        data = np.transpose(data, (2, 0, 1, 3))
    elif plane == "coronal":
        data = np.transpose(data, (1, 0, 2, 3))

    # all planes need to be rotated by 90 degrees in x-y to get the correct orientation
    data = np.rot90(data, k=1, axes=(1, 2))

    if not data_on_kspace:
        data = np.fft.ifftn(data, axes=(0, 1, 2))

    return data


def __compute_targets__(
    kspace: np.ndarray, coil_sensitivities: np.ndarray, coil_dim: int = -1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the target images from the kspace and coil sensitivities.

    Parameters
    ----------
    kspace : np.ndarray
        The kspace.
    coil_sensitivities : np.ndarray
        The coil sensitivities.
    coil_dim : int, optional
        The dimension of the coil sensitivities. The default is -1.

    Returns
    -------
    image_space : np.ndarray
        The image space.
    target_image : np.ndarray
        The target image.
    """
    # get the image space
    image_space = np.fft.fftshift(
        np.fft.ifftn(np.fft.fftshift(kspace, axes=(0, 1, 2)), axes=(0, 1, 2)), axes=(0, 1, 2)
    )

    # compute the target
    target = np.sum(image_space * np.conj(coil_sensitivities), axis=coil_dim)

    return image_space, target


def __compute_masks__(target_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the brain and head masks.

    Parameters
    ----------
    target_image : np.ndarray
        The target image.
    Returns
    -------
    brain_mask : np.ndarray
        The brain mask.
    head_mask : np.ndarray
        The head mask.
    """
    # compute head and brain mask
    head_masks = []
    brain_masks = []
    for _slice_idx_ in tqdm(range(target_image.shape[0])):
        # get the head mask
        head_mask = np.abs(target_image[_slice_idx_]) > threshold_otsu(np.abs(target_image[_slice_idx_]))
        # dilate the head mask
        head_mask = binary_dilation(head_mask, iterations=4)
        # get the convex hull of the head mask
        head_mask = convex_hull_image(head_mask)  # type: ignore
        head_masks.append(head_mask)
        brain_masks.append(1 - head_mask)
    head_mask = np.stack(head_masks, axis=0)
    brain_mask = np.stack(brain_masks, axis=0)
    return brain_mask.astype(np.float32), head_mask.astype(np.float32)


def __compute_quantitative_maps__(
    target_images: np.ndarray,
    TEs: List[float],
    brain_mask: np.ndarray,
    head_mask: np.ndarray,
    fully_sampled: bool = True,
    shift: bool = False,
    fft_centered: bool = False,
    fft_normalization: str = "backward",
    spatial_dims=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the quantitative maps.

    Parameters
    ----------
    target_images : np.ndarray
        The target images.
    TEs : List[float]
        The echo times.
    brain_mask : np.ndarray
        The brain mask.
    head_mask : np.ndarray
        The head mask.
    fully_sampled : bool, optional
        Whether the data is fully sampled or not. The default is True.
    shift : bool, optional
        Whether to shift the kspace or not. The default is False.
    fft_centered : bool, optional
        Whether the fft is centered or not. The default is False.
    fft_normalization : str, optional
        The fft normalization. The default is "backward".
    spatial_dims : List[int], optional
        The spatial dimensions. The default is [-2, -1].

    Returns
    -------
    multiple_echoes_target : np.ndarray
        The stacked target image from multiple echoes.
    R2star_map : np.ndarray
        The R2* map.
    S0_map : np.ndarray
        The S0 map.
    B0_map : np.ndarray
        The B0 map.
    phi_map : np.ndarray
        The phase map.
    """
    # stack real and imaginary part of the target image
    if spatial_dims is None:
        spatial_dims = [-2, -1]
    multiple_echoes_target_tensor = np.stack([np.real(target_images), np.imag(target_images)], axis=-1)
    # convert to torch tensor
    multiple_echoes_target_tensor = torch.from_numpy(multiple_echoes_target_tensor)
    # verify the tensor will be complex valued
    multiple_echoes_target_tensor = torch.view_as_complex(multiple_echoes_target_tensor)
    # verify the tensor can be converted to real valued, with stacked real and imag parts on the last dimension
    multiple_echoes_target_tensor = torch.view_as_real(multiple_echoes_target_tensor)

    brain_mask = torch.from_numpy(brain_mask).unsqueeze(1)
    head_mask = torch.from_numpy(head_mask).unsqueeze(1)

    R2star_maps = []
    S0_maps = []
    B0_maps = []
    phi_maps = []
    for slice_idx in tqdm(range(multiple_echoes_target_tensor.shape[0])):
        # compute the quantitative maps
        R2star_map, S0_map, B0_map, phi_map = R2star_B0_S0_phi_mapping(
            prediction=multiple_echoes_target_tensor[slice_idx],
            TEs=TEs,
            brain_mask=brain_mask[slice_idx],
            head_mask=head_mask[slice_idx],
            fully_sampled=fully_sampled,
            shift=shift,
            fft_centered=fft_centered,
            fft_normalization=fft_normalization,
            spatial_dims=spatial_dims,
        )
        R2star_maps.append(R2star_map)
        S0_maps.append(S0_map)
        B0_maps.append(B0_map[0])
        phi_maps.append(phi_map)

    R2star_maps = torch.stack(R2star_maps, dim=0).numpy()
    S0_maps = torch.stack(S0_maps, dim=0).numpy()
    B0_maps = torch.stack(B0_maps, dim=0).numpy()
    phi_maps = torch.stack(phi_maps, dim=0).numpy()

    return torch.view_as_complex(multiple_echoes_target_tensor).numpy(), R2star_maps, S0_maps, B0_maps, phi_maps


def __save_data__(
    image_space: np.ndarray,
    coil_sensitivities: np.ndarray,
    target: np.ndarray,
    mask_brain: np.ndarray,
    mask_head: np.ndarray,
    R2star_map: np.ndarray,
    S0_map: np.ndarray,
    B0_map: np.ndarray,
    phi_map: np.ndarray,
    output_path: Path,
    filename: str,
):
    """
    Save the data.

    Parameters
    ----------
    image_space : np.ndarray
        The image space.
    coil_sensitivities : np.ndarray
        The coil sensitivities.
    target : np.ndarray
        The target image.
    mask_brain : np.ndarray
        The brain mask.
    mask_head : np.ndarray
        The head mask.
    R2star_map : np.ndarray
        The R2* map.
    S0_map : np.ndarray
        The S0 map.
    B0_map : np.ndarray
        The B0 map.
    phi_map : np.ndarray
        The phase map.
    output_path : Path
        The output path.
    filename : str
        The filename.
    """
    # we need to move the coils dimension to dimension 2 and get kspace
    image_space = np.moveaxis(image_space, -1, 2)
    # we need to move the coils dimension to dimension 1 and get coil sensitivities
    coil_sensitivities = np.moveaxis(coil_sensitivities, -1, 1)

    # get kspace
    kspace = np.fft.fftn(image_space, axes=(-2, -1))
    kspace = np.fft.fftshift(kspace, axes=(-2, -1))

    if not os.path.exists(output_path):
        output_path.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path / f"{filename}.h5", "w") as f:
        f.create_dataset("kspace", data=kspace.astype(np.complex64))
        f.create_dataset("sensitivity_map", data=coil_sensitivities.astype(np.complex64))
        f.create_dataset("target", data=target.astype(np.complex64))
        f.create_dataset("mask_brain", data=mask_brain)
        f.create_dataset("mask_head", data=mask_head)
        f.create_dataset("R2star_map_target", data=R2star_map)
        f.create_dataset("S0_map_target", data=S0_map)
        f.create_dataset("B0_map_target", data=B0_map)
        f.create_dataset("phi_map_target", data=phi_map)


def main(args):
    # get all files
    files = list(Path(args.data_path).iterdir())
    # get the fnames
    fnames = [str(file).split("/")[-1].split("_")[1].split(".")[0] for file in files if "coilsens" in file.name]

    plane = args.plane

    # iterate over all subjects
    for fname in fnames:
        print(f"Processing subject {fname}...")

        # get all files for this subject from files
        subject_files = [file for file in files if fname in file.name]
        raw_data_files = [file for file in subject_files if "coilsens" not in file.name and "inv1" not in file.name]
        raw_data_files.sort()

        # preprocess the raw data
        kspaces = [__preprocess_ahead_raw_data__(str(x)) for x in raw_data_files]
        kspaces = [__get_plane__(x, data_on_kspace=True, plane=plane) for x in kspaces]

        # preprocess the coil sensitivities
        coil_sensitivities_file = [file for file in subject_files if "coilsens" in file.name][0]
        coil_sensitivities = __preprocess_ahead_coil_sensitivities__(str(coil_sensitivities_file))
        coil_sensitivities = __get_plane__(coil_sensitivities, data_on_kspace=False, plane=plane)

        # compute the image spaces and targets
        image_spaces = []
        targets = []
        for x in kspaces:
            image_space, target = __compute_targets__(x, coil_sensitivities, coil_dim=-1)
            image_spaces.append(image_space)
            targets.append(target)
        image_space = np.stack(image_spaces, axis=1)
        target = np.stack(targets, axis=1)

        # masks are the same for all echoes
        brain_mask, head_mask = __compute_masks__(targets[0])

        # compute the quantitative maps
        multiple_echoes_target_tensor, R2star_map, S0_map, B0_map, phi_map = __compute_quantitative_maps__(
            target,
            args.TEs,
            brain_mask,
            head_mask,
            fully_sampled=args.fully_sampled,
            shift=args.shift,
            fft_centered=args.fft_centered,
            fft_normalization=args.fft_normalization,
            spatial_dims=args.spatial_dims,
        )

        slice_range = args.slice_range
        if slice_range is not None:
            image_space = image_space[slice_range[0] : slice_range[1]]
            coil_sensitivities = coil_sensitivities[slice_range[0] : slice_range[1]]
            multiple_echoes_target_tensor = multiple_echoes_target_tensor[slice_range[0] : slice_range[1]]
            brain_mask = brain_mask[slice_range[0] : slice_range[1]]
            head_mask = head_mask[slice_range[0] : slice_range[1]]
            R2star_map = R2star_map[slice_range[0] : slice_range[1]]
            S0_map = S0_map[slice_range[0] : slice_range[1]]
            B0_map = B0_map[slice_range[0] : slice_range[1]]
            phi_map = phi_map[slice_range[0] : slice_range[1]]

        # save the data to disk
        __save_data__(
            image_space,
            coil_sensitivities,
            multiple_echoes_target_tensor,
            brain_mask,
            head_mask,
            R2star_map,
            S0_map,
            B0_map,
            phi_map,
            Path(args.output_path),
            f"mp2rageme_{fname}_{plane}",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=Path, default="data/ahead_data")
    parser.add_argument("--output_path", type=Path, default="data/ahead_data_preprocessed")
    parser.add_argument("--plane", type=str, default="axial")
    parser.add_argument("--slice_range", default=None, type=int, nargs="+")
    parser.add_argument("--TEs", type=float, nargs="+", default=[3.0, 11.5, 20.0, 28.5])
    parser.add_argument("--fully_sampled", type=bool, default=True)
    parser.add_argument("--shift", type=bool, default=False)
    parser.add_argument("--fft_centered", type=bool, default=False)
    parser.add_argument("--fft_normalization", type=str, default="backward")
    parser.add_argument("--spatial_dims", type=int, nargs="+", default=[-2, -1])
    args = parser.parse_args()
    main(args)
