# encoding: utf-8
__author__ = "Dimitrios Karkalousos, Chaoping Zhang"

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
        init_coil_dim: int = 0,
        fixed_precomputed_acceleration: Optional[int] = None,
        kspace_scaling_factor: float = 10000,
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
        init_coil_dim: The initial coil dimension of the data.
        fixed_precomputed_acceleration: Optional; A list of integers that determine the fixed acceleration of the
            data. If provided, the data will be loaded with the fixed acceleration.
        kspace_scaling_factor: A float that determines the scaling factor of the k-space data.
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
        self.init_coil_dim = init_coil_dim
        self.fixed_precomputed_acceleration = fixed_precomputed_acceleration
        self.kspace_scaling_factor = kspace_scaling_factor

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

        num_slices = 1 if data_saved_per_slice else shape[0]
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

    def check_stored_qdata(self, data, key, dataslice):
        """
        Check if quantitative data are stored in the dataset.

        Parameters
        ----------
        data: Data to extract slices from.
        key: Key to extract slices from.
        dataslice: Slice to extract.
        """
        qdata = []
        count = 0
        for k in data.keys():
            if key in k:
                acc = k.split("_")[-1].split("x")[0]
                if acc not in ["brain", "head"]:
                    x = self.get_consecutive_slices(data, key + str(acc) + "x", dataslice)
                    if x.ndim == 3:
                        x = x[dataslice]
                    if (
                        self.fixed_precomputed_acceleration is not None
                        and int(acc) == self.fixed_precomputed_acceleration
                        or self.fixed_precomputed_acceleration is None
                    ):
                        qdata.append(x)
                    else:
                        count += 1
            if self.fixed_precomputed_acceleration is not None:
                qdata = [qdata[0] * count]
        return qdata

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

            if self.init_coil_dim in [3, 4, -1]:
                kspace = np.transpose(kspace, (0, 3, 1, 2))  # [nr_TEs, nr_channels, nr_rows, nr_cols]

            kspace = kspace / self.kspace_scaling_factor

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

            if self.init_coil_dim in [3, 4, -1]:
                sensitivity_map = np.transpose(sensitivity_map, (2, 0, 1))  # [nr_channels, nr_rows, nr_cols]

            if "mask" in hf:
                mask = np.asarray(self.get_consecutive_slices(hf, "mask", dataslice))
                if mask.ndim == 3:
                    mask = mask[dataslice]
            elif any("mask_" in _ for _ in hf.keys()):
                mask = self.check_stored_qdata(hf, "mask_", dataslice)
            elif self.mask_root is not None and self.mask_root != "None":
                with h5py.File(Path(self.mask_root) / fname.name, "r") as mf:
                    mask = np.asarray(self.get_consecutive_slices(mf, "mask", dataslice))
            else:
                mask = np.empty([])

            if "mask_brain" in hf:
                mask_brain = np.asarray(self.get_consecutive_slices(hf, "mask_brain", dataslice))
            else:
                mask_brain = np.empty([])

            if "mask_head" in hf.keys():
                mask_head = np.asarray(self.get_consecutive_slices(hf, "mask_head", dataslice))
            else:
                mask_head = np.empty([])

            mask = [mask, mask_brain, mask_head]

            if any("B0_map_init_" in _ for _ in hf.keys()):
                B0_map = self.check_stored_qdata(hf, "B0_map_init_", dataslice)
                if all("B0_map_target" not in _ for _ in hf.keys()):
                    raise ValueError("While B0 map initializations are found, no B0 map target found in file.")
                B0_map_target = self.get_consecutive_slices(hf, "B0_map_target", dataslice)
                B0_map.append(B0_map_target)
            else:
                B0_map = np.empty([])

            if any("S0_map_init_" in _ for _ in hf.keys()):
                S0_map = self.check_stored_qdata(hf, "S0_map_init_", dataslice)
                if all("S0_map_target" not in _ for _ in hf.keys()):
                    raise ValueError("While S0 map initializations are found, no S0 map target found in file.")
                S0_map_target = self.get_consecutive_slices(hf, "S0_map_target", dataslice)
                S0_map.append(S0_map_target)
            else:
                S0_map = np.empty([])

            if any("R2star_map_init_" in _ for _ in hf.keys()):
                R2star_map = self.check_stored_qdata(hf, "R2star_map_init_", dataslice)
                if all("R2star_map_target" not in _ for _ in hf.keys()):
                    raise ValueError("While R2star map initializations are found, no R2star map target found in file.")
                R2star_map_target = self.get_consecutive_slices(hf, "R2star_map_target", dataslice)
                R2star_map.append(R2star_map_target)
            else:
                R2star_map = np.empty([])

            if any("phi_map_init_" in _ for _ in hf.keys()):
                phi_map = self.check_stored_qdata(hf, "phi_map_init_", dataslice)
                if all("phi_map_target" not in _ for _ in hf.keys()):
                    raise ValueError("While phi map initializations are found, no phi map target found in file.")
                phi_map_target = self.get_consecutive_slices(hf, "phi_map_target", dataslice)
                phi_map.append(phi_map_target)
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
