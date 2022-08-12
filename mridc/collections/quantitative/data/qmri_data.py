# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

import logging
import os
import random
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import h5py
import numpy as np
import yaml  # type: ignore
from torch.utils.data import Dataset

from mridc.collections.common.parts.utils import is_none


class qMRISliceDataset(Dataset):
    """A dataset that loads slices from a single dataset."""

    def __init__(
        self,
        root: Union[str, Path, os.PathLike],
        transform: Optional[Callable] = None,
        sense_root: Union[str, Path, os.PathLike] = None,
        sequence: str = None,
        use_dataset_cache: bool = False,
        sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
        dataset_cache_file: Union[str, Path, os.PathLike] = "dataset_cache.yaml",
        num_cols: Optional[Tuple[int]] = None,
        mask_root: Union[str, Path, os.PathLike] = None,
        consecutive_slices: int = 1,
        data_saved_per_slice: bool = False,
        fixed_precomputed_acceleration: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        root: Path to the dataset.
        transform: Optional; A sequence of callable objects that preprocesses the raw data into appropriate form.
            The transform function should take 'kspace', 'target', 'attributes', 'filename', and 'slice' as inputs.
            'target' may be null for test data.
        sense_root: Path to the coil sensitivities maps dataset.
        sequence: Sequence of the dataset.
        use_dataset_cache: Whether to cache dataset metadata. This is very useful for large datasets like the brain
            data.
        sample_rate: Optional; A sequence of floats between 0 and 1. This controls what fraction of the slices
            should be loaded. When creating subsampled datasets either set sample_rates (sample by slices) or
            volume_sample_rates (sample by volumes) but not both.
        volume_sample_rate: Optional; A sequence of floats between 0 and 1. This controls what fraction of the
             volumes should be loaded. When creating subsampled datasets either set sample_rates (sample by slices)
              or volume_sample_rates (sample by volumes) but not both.
        dataset_cache_file: Optional; A file in which to cache dataset information for faster load times.
        num_cols: Optional; If provided, only slices with the desired number of columns will be considered.
        mask_root: Path to stored masks.
        consecutive_slices: An int (>0) that determine the amount of consecutive slices of the file to be loaded at
            the same time. Defaults to 1, loading single slices.
        data_saved_per_slice: Whether the data is saved per slice or per volume.
        fixed_precomputed_acceleration: Optional; A list of integers that determine the fixed acceleration of the
            data. If provided, the data will be loaded with the fixed acceleration.
        """
        if sequence not in ("MEGRE", "FUTURE_SEQUENCES"):
            raise ValueError(f'Sequence should be either "MEGRE" or "FUTURE_SEQUENCES". Found {sequence}.')

        if sample_rate is not None and volume_sample_rate is not None:
            raise ValueError(
                "either set sample_rate (sample by slices) or volume_sample_rate (sample by volumes) but not both"
            )

        self.sense_root = sense_root
        self.mask_root = mask_root

        self.dataset_cache_file = Path(dataset_cache_file)

        self.data_saved_per_slice = data_saved_per_slice
        self.fixed_precomputed_acceleration = fixed_precomputed_acceleration

        self.transform = transform
        self.recons_key = "reconstruction"
        self.examples = []

        # set default sampling mode if none given
        if sample_rate is None:
            sample_rate = 1.0
        if volume_sample_rate is None:
            volume_sample_rate = 1.0

        # load dataset cache if we have and user wants to use it
        if self.dataset_cache_file.exists() and use_dataset_cache:
            with open(self.dataset_cache_file, "rb") as f:
                dataset_cache = yaml.safe_load(f)
        else:
            dataset_cache = {}

        # check if our dataset is in the cache
        # if there, use that metadata, if not, then regenerate the metadata
        if dataset_cache.get(root) is None or not use_dataset_cache:
            files = list(Path(root).iterdir())
            for fname in sorted(files):
                metadata, num_slices = self._retrieve_metadata(fname, data_saved_per_slice=self.data_saved_per_slice)
                if not is_none(num_slices) and not is_none(consecutive_slices):
                    num_slices = num_slices - (consecutive_slices - 1)
                self.examples += [(fname, slice_ind, metadata) for slice_ind in range(num_slices)]

            if dataset_cache.get(root) is None and use_dataset_cache:
                dataset_cache[root] = self.examples
                logging.info(f"Saving dataset cache to {self.dataset_cache_file}.")
                with open(self.dataset_cache_file, "wb") as f:  # type: ignore
                    yaml.dump(dataset_cache, f)  # type: ignore
        else:
            logging.info(f"Using dataset cache from {self.dataset_cache_file}.")
            self.examples = dataset_cache[root]

        # subsample if desired
        if sample_rate < 1.0:  # sample by slice
            random.shuffle(self.examples)
            num_examples = round(len(self.examples) * sample_rate)
            self.examples = self.examples[:num_examples]
        elif volume_sample_rate < 1.0:  # sample by volume
            vol_names = sorted(list({f[0].stem for f in self.examples}))
            random.shuffle(vol_names)
            num_volumes = round(len(vol_names) * volume_sample_rate)
            sampled_vols = vol_names[:num_volumes]
            self.examples = [example for example in self.examples if example[0].stem in sampled_vols]

        if num_cols:
            self.examples = [ex for ex in self.examples if ex[2]["encoding_size"][1] in num_cols]  # type: ignore

        # Create random number generator used for consecutive slice selection and set consecutive slice amount
        self.consecutive_slices = consecutive_slices
        if self.consecutive_slices < 1:
            raise ValueError("consecutive_slices value is out of range, must be > 0.")

    @staticmethod
    def _retrieve_metadata(fname, data_saved_per_slice=False):
        """
        Retrieve metadata from a given file.

        Parameters
        ----------
        fname: Path to file.
        data_saved_per_slice: Whether the data is saved per slice or per volume.

        Returns
        -------
        A dictionary containing the metadata.
        """
        with h5py.File(fname, "r") as hf:
            padding_left = 0
            padding_right = 0
            enc_size = 0
            recon_size = (0, 0)

            if "kspace" in hf:
                shape = hf["kspace"].shape
            elif "ksp" in hf:
                shape = hf["ksp"].shape
            elif "reconstruction" in hf:
                shape = hf["reconstruction"].shape
            else:
                raise ValueError(f"{fname} does not contain kspace or reconstruction data.")

        if data_saved_per_slice:
            num_slices = 1
        else:
            num_slices = shape[0]

        metadata = {
            "padding_left": padding_left,
            "padding_right": padding_right,
            "encoding_size": enc_size,
            "recon_size": recon_size,
        }

        return metadata, num_slices

    def get_consecutive_slices(self, data, key, dataslice):
        """
        Get consecutive slices from a given data.
        Args:
            data: Data to extract slices from.
            key: Key to extract slices from.
            dataslice: Slice to extract slices from.
        Returns:
            A list of consecutive slices.
        """
        data = data[key]

        if self.data_saved_per_slice:
            data = np.expand_dims(data, axis=0)

        if self.consecutive_slices == 1:
            if data.shape[0] == 1:
                return data[0]
            elif data.ndim != 2:
                return data[dataslice]
            return data

        num_slices = data.shape[0]
        if self.consecutive_slices > num_slices:
            return np.stack(data, axis=0)

        start_slice = dataslice
        if dataslice + self.consecutive_slices <= num_slices:
            end_slice = dataslice + self.consecutive_slices
        else:
            end_slice = num_slices

        return data[start_slice:end_slice]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i: int):
        fname, dataslice, metadata = self.examples[i]
        with h5py.File(fname, "r") as hf:
            if "kspace" in hf:
                kspace = self.get_consecutive_slices(hf, "kspace", dataslice).astype(np.complex64)
            elif "ksp" in hf:
                kspace = self.get_consecutive_slices(hf, "ksp", dataslice).astype(np.complex64)
            else:
                raise ValueError("No kspace data found in file. Only 'kspace' or 'ksp' keys are supported.")

            kspace = np.transpose(kspace, (0, 3, 1, 2))  # [nr_TEs, nr_channels, nr_rows, nr_cols]
            kspace = kspace / 10000

            if "sensitivity_map" in hf:
                sensitivity_map = self.get_consecutive_slices(hf, "sensitivity_map", dataslice).astype(np.complex64)
            elif "sense" in hf:
                sensitivity_map = self.get_consecutive_slices(hf, "sense", dataslice).astype(np.complex64)
            elif self.sense_root is not None and self.sense_root != "None":
                with h5py.File(Path(self.sense_root) / Path(str(fname).split("/")[-2]) / fname.name, "r") as sf:
                    if "sensitivity_map" in sf or "sensitivity_map" in next(iter(sf.keys())):
                        sensitivity_map = self.get_consecutive_slices(sf, "sensitivity_map", dataslice)
                    else:
                        sensitivity_map = self.get_consecutive_slices(sf, "sense", dataslice)
                    sensitivity_map = sensitivity_map.squeeze().astype(np.complex64)
            else:
                sensitivity_map = np.array([])

            sensitivity_map = np.transpose(sensitivity_map, (2, 0, 1))  # [nr_TEs, nr_channels, nr_rows, nr_cols]

            if "mask" in hf:
                mask = np.asarray(self.get_consecutive_slices(hf, "mask", dataslice))
                if mask.ndim == 3:
                    mask = mask[dataslice]
            elif "mask_3x" in hf:
                mask_3x = self.get_consecutive_slices(hf, "mask_3x", dataslice)
                if mask_3x.ndim == 3:
                    mask_3x = mask_3x[dataslice]
                mask_6x = self.get_consecutive_slices(hf, "mask_6x", dataslice)
                if mask_6x.ndim == 3:
                    mask_6x = mask_6x[dataslice]
                mask_9x = self.get_consecutive_slices(hf, "mask_9x", dataslice)
                if mask_9x.ndim == 3:
                    mask_9x = mask_9x[dataslice]
                mask_12x = self.get_consecutive_slices(hf, "mask_12x", dataslice)
                if mask_12x.ndim == 3:
                    mask_12x = mask_12x[dataslice]

                if self.fixed_precomputed_acceleration is not None:
                    if self.fixed_precomputed_acceleration == 3:
                        mask_6x = mask_3x
                        mask_9x = mask_3x
                        mask_12x = mask_3x
                    elif self.fixed_precomputed_acceleration == 6:
                        mask_3x = mask_6x
                        mask_9x = mask_6x
                        mask_12x = mask_6x
                    elif self.fixed_precomputed_acceleration == 9:
                        mask_3x = mask_9x
                        mask_6x = mask_9x
                        mask_12x = mask_9x
                    elif self.fixed_precomputed_acceleration == 12:
                        mask_3x = mask_12x
                        mask_6x = mask_12x
                        mask_9x = mask_12x
                    else:
                        raise ValueError(
                            f"{self.fixed_precomputed_acceleration}x is not a valid precomputed acceleration factor."
                        )
                mask = [mask_3x, mask_6x, mask_9x, mask_12x]
            elif self.mask_root is not None and self.mask_root != "None":
                with h5py.File(Path(self.mask_root) / fname.name, "r") as mf:
                    mask = np.asarray(self.get_consecutive_slices(mf, "mask", dataslice))
            else:
                mask = np.empty([])

            if "mask_brain" in hf:
                mask_brain = np.asarray(self.get_consecutive_slices(hf, "mask_brain", dataslice))
            else:
                mask_brain = np.empty([])

            if "mask_head" in hf:
                mask_head = np.asarray(self.get_consecutive_slices(hf, "mask_head", dataslice))
            else:
                mask_head = np.empty([])

            mask = [mask, mask_brain, mask_head]

            if "B0_map_init_12x" in hf:
                B0_map_init_3x = self.get_consecutive_slices(hf, "B0_map_init_3x", dataslice)
                B0_map_init_6x = self.get_consecutive_slices(hf, "B0_map_init_6x", dataslice)
                B0_map_init_9x = self.get_consecutive_slices(hf, "B0_map_init_9x", dataslice)
                B0_map_init_12x = self.get_consecutive_slices(hf, "B0_map_init_12x", dataslice)
                B0_map_target = self.get_consecutive_slices(hf, "B0_map_target", dataslice)
                B0_map_recon_3x = self.get_consecutive_slices(hf, "B0_map_recon_3x", dataslice)
                B0_map_recon_6x = self.get_consecutive_slices(hf, "B0_map_recon_6x", dataslice)
                B0_map_recon_9x = self.get_consecutive_slices(hf, "B0_map_recon_9x", dataslice)
                B0_map_recon_12x = self.get_consecutive_slices(hf, "B0_map_recon_12x", dataslice)

                if self.fixed_precomputed_acceleration is not None:
                    if self.fixed_precomputed_acceleration == 3:
                        B0_map_init_6x = B0_map_init_3x
                        B0_map_init_9x = B0_map_init_3x
                        B0_map_init_12x = B0_map_init_3x
                        B0_map_recon_6x = B0_map_recon_3x
                        B0_map_recon_9x = B0_map_recon_3x
                        B0_map_recon_12x = B0_map_recon_3x
                    elif self.fixed_precomputed_acceleration == 6:
                        B0_map_init_3x = B0_map_init_6x
                        B0_map_init_9x = B0_map_init_6x
                        B0_map_init_12x = B0_map_init_6x
                        B0_map_recon_3x = B0_map_recon_6x
                        B0_map_recon_9x = B0_map_recon_6x
                        B0_map_recon_12x = B0_map_recon_6x
                    elif self.fixed_precomputed_acceleration == 9:
                        B0_map_init_3x = B0_map_init_9x
                        B0_map_init_6x = B0_map_init_9x
                        B0_map_init_12x = B0_map_init_9x
                        B0_map_recon_3x = B0_map_recon_9x
                        B0_map_recon_6x = B0_map_recon_9x
                        B0_map_recon_12x = B0_map_recon_9x
                    elif self.fixed_precomputed_acceleration == 12:
                        B0_map_init_3x = B0_map_init_12x
                        B0_map_init_6x = B0_map_init_12x
                        B0_map_init_9x = B0_map_init_12x
                        B0_map_recon_3x = B0_map_recon_12x
                        B0_map_recon_6x = B0_map_recon_12x
                        B0_map_recon_9x = B0_map_recon_12x
                    else:
                        raise ValueError(
                            f"{self.fixed_precomputed_acceleration}x is not a valid precomputed acceleration factor."
                        )
                B0_map = [
                    B0_map_init_3x,
                    B0_map_init_6x,
                    B0_map_init_9x,
                    B0_map_init_12x,
                    B0_map_target,
                    B0_map_recon_3x,
                    B0_map_recon_6x,
                    B0_map_recon_9x,
                    B0_map_recon_12x,
                ]
            else:
                B0_map = np.empty([])

            if "S0_map_init_12x" in hf:
                S0_map_init_3x = self.get_consecutive_slices(hf, "S0_map_init_3x", dataslice)
                S0_map_init_6x = self.get_consecutive_slices(hf, "S0_map_init_6x", dataslice)
                S0_map_init_9x = self.get_consecutive_slices(hf, "S0_map_init_9x", dataslice)
                S0_map_init_12x = self.get_consecutive_slices(hf, "S0_map_init_12x", dataslice)
                S0_map_target = self.get_consecutive_slices(hf, "S0_map_target", dataslice)
                S0_map_recon_3x = self.get_consecutive_slices(hf, "S0_map_recon_3x", dataslice)
                S0_map_recon_6x = self.get_consecutive_slices(hf, "S0_map_recon_6x", dataslice)
                S0_map_recon_9x = self.get_consecutive_slices(hf, "S0_map_recon_9x", dataslice)
                S0_map_recon_12x = self.get_consecutive_slices(hf, "S0_map_recon_12x", dataslice)
                if self.fixed_precomputed_acceleration is not None:
                    if self.fixed_precomputed_acceleration == 3:
                        S0_map_init_6x = S0_map_init_3x
                        S0_map_init_9x = S0_map_init_3x
                        S0_map_init_12x = S0_map_init_3x
                        S0_map_recon_6x = S0_map_recon_3x
                        S0_map_recon_9x = S0_map_recon_3x
                        S0_map_recon_12x = S0_map_recon_3x
                    elif self.fixed_precomputed_acceleration == 6:
                        S0_map_init_3x = S0_map_init_6x
                        S0_map_init_9x = S0_map_init_6x
                        S0_map_init_12x = S0_map_init_6x
                        S0_map_recon_3x = S0_map_recon_6x
                        S0_map_recon_9x = S0_map_recon_6x
                        S0_map_recon_12x = S0_map_recon_6x
                    elif self.fixed_precomputed_acceleration == 9:
                        S0_map_init_3x = S0_map_init_9x
                        S0_map_init_6x = S0_map_init_9x
                        S0_map_init_12x = S0_map_init_9x
                        S0_map_recon_3x = S0_map_recon_9x
                        S0_map_recon_6x = S0_map_recon_9x
                        S0_map_recon_12x = S0_map_recon_9x
                    elif self.fixed_precomputed_acceleration == 12:
                        S0_map_init_3x = S0_map_init_12x
                        S0_map_init_6x = S0_map_init_12x
                        S0_map_init_9x = S0_map_init_12x
                        S0_map_recon_3x = S0_map_recon_12x
                        S0_map_recon_6x = S0_map_recon_12x
                        S0_map_recon_9x = S0_map_recon_12x
                    else:
                        raise ValueError(
                            f"{self.fixed_precomputed_acceleration}x is not a valid precomputed acceleration factor."
                        )
                S0_map = [
                    S0_map_init_3x,
                    S0_map_init_6x,
                    S0_map_init_9x,
                    S0_map_init_12x,
                    S0_map_target,
                    S0_map_recon_3x,
                    S0_map_recon_6x,
                    S0_map_recon_9x,
                    S0_map_recon_12x,
                ]
            else:
                S0_map = np.empty([])

            if "R2star_map_init_12x" in hf:
                R2star_map_init_3x = self.get_consecutive_slices(hf, "R2star_map_init_3x", dataslice)
                R2star_map_init_6x = self.get_consecutive_slices(hf, "R2star_map_init_6x", dataslice)
                R2star_map_init_9x = self.get_consecutive_slices(hf, "R2star_map_init_9x", dataslice)
                R2star_map_init_12x = self.get_consecutive_slices(hf, "R2star_map_init_12x", dataslice)
                R2star_map_target = self.get_consecutive_slices(hf, "R2star_map_target", dataslice)
                R2star_map_recon_3x = self.get_consecutive_slices(hf, "R2star_map_recon_3x", dataslice)
                R2star_map_recon_6x = self.get_consecutive_slices(hf, "R2star_map_recon_6x", dataslice)
                R2star_map_recon_9x = self.get_consecutive_slices(hf, "R2star_map_recon_9x", dataslice)
                R2star_map_recon_12x = self.get_consecutive_slices(hf, "R2star_map_recon_12x", dataslice)

                if self.fixed_precomputed_acceleration is not None:
                    if self.fixed_precomputed_acceleration == 3:
                        R2star_map_init_6x = R2star_map_init_3x
                        R2star_map_init_9x = R2star_map_init_3x
                        R2star_map_init_12x = R2star_map_init_3x
                        R2star_map_recon_6x = R2star_map_recon_3x
                        R2star_map_recon_9x = R2star_map_recon_3x
                        R2star_map_recon_12x = R2star_map_recon_3x
                    elif self.fixed_precomputed_acceleration == 6:
                        R2star_map_init_3x = R2star_map_init_6x
                        R2star_map_init_9x = R2star_map_init_6x
                        R2star_map_init_12x = R2star_map_init_6x
                        R2star_map_recon_3x = R2star_map_recon_6x
                        R2star_map_recon_9x = R2star_map_recon_6x
                        R2star_map_recon_12x = R2star_map_recon_6x
                    elif self.fixed_precomputed_acceleration == 9:
                        R2star_map_init_3x = R2star_map_init_9x
                        R2star_map_init_6x = R2star_map_init_9x
                        R2star_map_init_12x = R2star_map_init_9x
                        R2star_map_recon_3x = R2star_map_recon_9x
                        R2star_map_recon_6x = R2star_map_recon_9x
                        R2star_map_recon_12x = R2star_map_recon_9x
                    elif self.fixed_precomputed_acceleration == 12:
                        R2star_map_init_3x = R2star_map_init_12x
                        R2star_map_init_6x = R2star_map_init_12x
                        R2star_map_init_9x = R2star_map_init_12x
                        R2star_map_recon_3x = R2star_map_recon_12x
                        R2star_map_recon_6x = R2star_map_recon_12x
                        R2star_map_recon_9x = R2star_map_recon_12x
                    else:
                        raise ValueError(
                            f"{self.fixed_precomputed_acceleration}x is not a valid precomputed acceleration factor."
                        )
                R2star_map = [
                    R2star_map_init_3x,
                    R2star_map_init_6x,
                    R2star_map_init_9x,
                    R2star_map_init_12x,
                    R2star_map_target,
                    R2star_map_recon_3x,
                    R2star_map_recon_6x,
                    R2star_map_recon_9x,
                    R2star_map_recon_12x,
                ]
            else:
                R2star_map = np.empty([])

            if "phi_map_init_12x" in hf:
                phi_map_init_3x = self.get_consecutive_slices(hf, "phi_map_init_3x", dataslice)
                phi_map_init_6x = self.get_consecutive_slices(hf, "phi_map_init_6x", dataslice)
                phi_map_init_9x = self.get_consecutive_slices(hf, "phi_map_init_9x", dataslice)
                phi_map_init_12x = self.get_consecutive_slices(hf, "phi_map_init_12x", dataslice)
                phi_map_target = self.get_consecutive_slices(hf, "phi_map_target", dataslice)
                phi_map_recon_3x = self.get_consecutive_slices(hf, "phi_map_recon_3x", dataslice)
                phi_map_recon_6x = self.get_consecutive_slices(hf, "phi_map_recon_6x", dataslice)
                phi_map_recon_9x = self.get_consecutive_slices(hf, "phi_map_recon_9x", dataslice)
                phi_map_recon_12x = self.get_consecutive_slices(hf, "phi_map_recon_12x", dataslice)
                if self.fixed_precomputed_acceleration is not None:
                    if self.fixed_precomputed_acceleration == 3:
                        phi_map_init_6x = phi_map_init_3x
                        phi_map_init_9x = phi_map_init_3x
                        phi_map_init_12x = phi_map_init_3x
                        phi_map_recon_6x = phi_map_recon_3x
                        phi_map_recon_9x = phi_map_recon_3x
                        phi_map_recon_12x = phi_map_recon_3x
                    elif self.fixed_precomputed_acceleration == 6:
                        phi_map_init_3x = phi_map_init_6x
                        phi_map_init_9x = phi_map_init_6x
                        phi_map_init_12x = phi_map_init_6x
                        phi_map_recon_3x = phi_map_recon_6x
                        phi_map_recon_9x = phi_map_recon_6x
                        phi_map_recon_12x = phi_map_recon_6x
                    elif self.fixed_precomputed_acceleration == 9:
                        phi_map_init_3x = phi_map_init_9x
                        phi_map_init_6x = phi_map_init_9x
                        phi_map_init_12x = phi_map_init_9x
                        phi_map_recon_3x = phi_map_recon_9x
                        phi_map_recon_6x = phi_map_recon_9x
                        phi_map_recon_12x = phi_map_recon_9x
                    elif self.fixed_precomputed_acceleration == 12:
                        phi_map_init_3x = phi_map_init_12x
                        phi_map_init_6x = phi_map_init_12x
                        phi_map_init_9x = phi_map_init_12x
                        phi_map_recon_3x = phi_map_recon_12x
                        phi_map_recon_6x = phi_map_recon_12x
                        phi_map_recon_9x = phi_map_recon_12x
                    else:
                        raise ValueError(
                            f"{self.fixed_precomputed_acceleration}x is not a valid precomputed acceleration factor."
                        )
                phi_map = [
                    phi_map_init_3x,
                    phi_map_init_6x,
                    phi_map_init_9x,
                    phi_map_init_12x,
                    phi_map_target,
                    phi_map_recon_3x,
                    phi_map_recon_6x,
                    phi_map_recon_9x,
                    phi_map_recon_12x,
                ]
            else:
                phi_map = np.empty([])

            qmaps = [B0_map, S0_map, R2star_map, phi_map]

            eta = (
                self.get_consecutive_slices(hf, "eta", dataslice).astype(np.complex64) if "eta" in hf else np.array([])
            )

            if "reconstruction_sense" in hf:
                self.recons_key = "reconstruction_sense"

            target = self.get_consecutive_slices(hf, self.recons_key, dataslice) if self.recons_key in hf else None

            attrs = dict(hf.attrs)
            attrs.update(metadata)

        if self.data_saved_per_slice:
            # arbitrary slice number for logging purposes
            fname = fname.name  # type: ignore
            dataslice = int(fname.split("_")[-1])  # type: ignore
            fname = "_".join(fname.split("_")[:-1])  # type: ignore

        return (
            (
                kspace,
                sensitivity_map,
                qmaps,
                mask,
                eta,
                target,
                attrs,
                fname,
                dataslice,
            )
            if self.transform is None
            else self.transform(
                kspace,
                sensitivity_map,
                qmaps,
                mask,
                eta,
                target,
                attrs,
                fname,
                dataslice,
            )
        )
