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


class JRSMRISliceDataset(Dataset):
    def __init__(
        self,
        root: Union[str, Path, os.PathLike],
        sense_root: Union[str, Path, os.PathLike] = None,
        mask_root: Union[str, Path, os.PathLike] = None,
        segmentations_root: Union[str, Path, os.PathLike] = None,
        sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
        use_dataset_cache: bool = False,
        dataset_cache_file: Union[str, Path, os.PathLike] = "dataset_cache.yaml",
        num_cols: Optional[Tuple[int]] = None,
        consecutive_slices: int = 1,
        segmentation_classes: int = 2,
        complex_data: bool = True,
        data_saved_per_slice: bool = False,
        transform: Optional[Callable] = None,
    ):
        """
        Joint Reconstruction & Segmentation MRI Dataset.

        Parameters
        ----------
        root: str
            Path to the dataset.
        sense_root: str
            Path to the sensitivity maps, if are stored separately.
        mask_root: str
            Path to the masks, if are stored separately.
        segmentations_root: str
            Path to the segmentations, if are stored separately.
        sample_rate: float
            Sample rate of the dataset.
        volume_sample_rate: float
            Sample rate of the volumes.
        use_dataset_cache: bool
            Use dataset cache.
        dataset_cache_file: str
            Path to the dataset cache file.
        num_cols: tuple
            Number of columns to use.
        consecutive_slices: int
            Number of consecutive slices to use.
        segmentation_classes: int
            Number of segmentation classes.
        complex_data: bool
            Use complex data.
        data_saved_per_slice: bool
            If the data are saved per slice.
        transform: callable
            Transform to apply to the data.
        """
        if not is_none(sample_rate) and not is_none(volume_sample_rate):
            raise ValueError(
                "either set sample_rate (sample by slices) or volume_sample_rate (sample by volumes) but not both"
            )

        self.sense_root = sense_root
        self.mask_root = mask_root
        self.segmentations_root = segmentations_root

        # set default sampling mode if none given
        if is_none(sample_rate):
            sample_rate = 1.0
        if is_none(volume_sample_rate):
            volume_sample_rate = 1.0

        self.dataset_cache_file = Path(dataset_cache_file)

        # load dataset cache if exists and user wants to use it
        if self.dataset_cache_file.exists() and use_dataset_cache:
            with open(self.dataset_cache_file, "rb") as f:
                dataset_cache = yaml.safe_load(f)
        else:
            dataset_cache = {}

        self.data_saved_per_slice = data_saved_per_slice
        self.examples = []
        # Check if the dataset is in the cache. f true, use that metadata. If false, then regenerate the metadata.
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
        if sample_rate < 1.0:  # type: ignore
            random.shuffle(self.examples)
            num_examples = round(len(self.examples) * sample_rate)  # type: ignore
            self.examples = self.examples[:num_examples]
        elif volume_sample_rate < 1.0:  # type: ignore
            vol_names = sorted(list({f[0].stem for f in self.examples}))
            random.shuffle(vol_names)
            num_volumes = round(len(vol_names) * volume_sample_rate)  # type: ignore
            sampled_vols = vol_names[:num_volumes]
            self.examples = [example for example in self.examples if example[0].stem in sampled_vols]

        if not is_none(num_cols):
            self.examples = [ex for ex in self.examples if ex[2]["encoding_size"][1] in num_cols]  # type: ignore

        # Create random number generator used for consecutive slice selection and set consecutive slice amount
        self.consecutive_slices = consecutive_slices
        if self.consecutive_slices < 1:
            raise ValueError("consecutive_slices value is out of range, must be > 0.")
        self.segmentation_classes = segmentation_classes
        self.complex_data = complex_data
        self.transform = transform

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

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i: int):
        fname, dataslice, metadata = self.examples[i]
        with h5py.File(fname, "r") as hf:
            if self.complex_data:
                if "kspace" in hf:
                    kspace = self.get_consecutive_slices(hf, "kspace", dataslice).astype(np.complex64)
                elif "ksp" in hf:
                    kspace = self.get_consecutive_slices(hf, "ksp", dataslice).astype(np.complex64)
                else:
                    raise ValueError(
                        "Complex data has been selected but no kspace data found in file. "
                        "Only 'kspace' or 'ksp' keys are supported."
                    )

                if "sensitivity_map" in hf:
                    sensitivity_map = self.get_consecutive_slices(hf, "sensitivity_map", dataslice).astype(
                        np.complex64
                    )
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

                if "mask" in hf:
                    mask = np.asarray(self.get_consecutive_slices(hf, "mask", dataslice))
                    if mask.ndim == 3:
                        mask = mask[dataslice]
                elif self.mask_root is not None and self.mask_root != "None":
                    with h5py.File(Path(self.mask_root) / fname.name, "r") as mf:
                        mask = np.asarray(self.get_consecutive_slices(mf, "mask", dataslice))
                else:
                    mask = np.empty([])
                imspace = np.empty([])
            elif not self.complex_data:
                if "reconstruction" in hf:
                    imspace = self.get_consecutive_slices(hf, "reconstruction", dataslice).astype(np.float32)
                else:
                    raise ValueError(
                        "Complex data has not been selected but no reconstruction data found in file. "
                        "Only 'reconstruction' key is supported."
                    )
                kspace = np.empty([])
                sensitivity_map = np.array([])
                mask = np.empty([])

            if "segmentation" in hf:
                segmentation_labels = np.asarray(self.get_consecutive_slices(hf, "segmentation", dataslice))
                if segmentation_labels.shape[-1] == self.segmentation_classes:
                    if segmentation_labels.ndim == 3:
                        segmentation_labels = np.transpose(segmentation_labels, (2, 0, 1))
                    else:
                        segmentation_labels = np.transpose(segmentation_labels, (0, 3, 1, 2))
                if segmentation_labels.ndim == 2:
                    segmentation_labels = np.expand_dims(segmentation_labels, axis=0)
            elif self.segmentations_root is not None and self.segmentations_root != "None":
                with h5py.File(Path(self.segmentations_root) / fname.name, "r") as sf:
                    segmentation_labels = np.asarray(self.get_consecutive_slices(sf, "segmentation", dataslice))
                    if segmentation_labels.shape[-1] == self.segmentation_classes:
                        if segmentation_labels.ndim == 3:
                            segmentation_labels = np.transpose(segmentation_labels, (2, 0, 1))
                        else:
                            segmentation_labels = np.transpose(segmentation_labels, (0, 3, 1, 2))
                    if segmentation_labels.ndim == 2:
                        segmentation_labels = np.expand_dims(segmentation_labels, axis=0)
            else:
                segmentation_labels = np.empty([])

            attrs = dict(hf.attrs)
            attrs.update(metadata)

        return (
            (
                kspace,
                imspace,
                sensitivity_map,
                mask,
                segmentation_labels,
                attrs,
                fname.name,
                dataslice,
            )
            if self.transform is None
            else self.transform(
                kspace,
                imspace,
                sensitivity_map,
                mask,
                segmentation_labels,
                attrs,
                fname.name,
                dataslice,
            )
        )
