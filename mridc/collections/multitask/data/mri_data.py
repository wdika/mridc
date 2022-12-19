# coding=utf-8
__author__ = "Dimitrios Karkalousos"

import json
import logging
import os
import random
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import h5py
import numpy as np
import nibabel as nib
import yaml  # type: ignore
import zarr
from torch.utils.data import Dataset

from mridc.collections.common.parts.utils import apply_mask, is_none


class SKMTEADataset(Dataset):
    def __init__(
            self,
            root: Union[str, Path, os.PathLike],
            mask_root: Union[str, Path, os.PathLike] = None,
            segmentations_root: Union[str, Path, os.PathLike] = None,
            annotations_root: Union[str, Path, os.PathLike] = None,
            initial_predictions_root: Union[str, Path, os.PathLike] = None,
            split: str = "train",
            sample_rate: Optional[float] = None,
            volume_sample_rate: Optional[float] = None,
            use_dataset_cache: bool = False,
            dataset_cache_file: Union[str, Path, os.PathLike] = "dataset_cache.yaml",
            num_cols: Optional[Tuple[int]] = None,
            consecutive_slices: int = 1,
            segmentation_classes: int = 4,
            complex_data: bool = True,
            data_saved_per_slice: bool = False,
            transform: Optional[Callable] = None,
    ):
        """
        SKM-TEA MRI Dataset.

        Parameters
        ----------
        root: str
            Path to the dataset.
        mask_root: str
            Path to the masks, if are stored separately.
        segmentations_root: str
            Path to the segmentations, if are stored separately.
        annotations_root: str
            Path to the annotations_root, if are stored separately.
        initial_predictions_root: str
            Path to the initial predictions, if there are any.
        split: str
            Split to use. One of "train", "val", "test".
        plane: str
            Plane to use. One of "axial", "coronal", "sagittal", "all".
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

        self.root = Path(root)
        self.mask_root = Path(mask_root) if mask_root is not None else None
        self.segmentations_root = (Path(segmentations_root) if segmentations_root is not None else None)

        with open(f"{annotations_root}/{split}.json") as f:
            annotations_file = json.load(f)
        self.split_files = [f["file_name"] for f in annotations_file["images"]]

        self.initial_predictions_root = (
            Path(initial_predictions_root) if initial_predictions_root is not None else None
        )

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
        # Check if the dataset is in the cache. If true, use that metadata. If false, then regenerate the metadata.
        if dataset_cache.get(root) is None or not use_dataset_cache:
            files = list(Path(root).iterdir())
            # keep the files that are in the split json file
            files = [f for f in files if f.name in self.split_files]
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
            if data.ndim != 2:
                if dataslice > data.shape[0] - 1:
                    dataslice = np.random.randint(0, data.shape[0] - 1)
                return data[dataslice]
            return data

        num_slices = data.shape[0]
        if self.consecutive_slices > num_slices or self.consecutive_slices == -1:
            return np.stack(data, axis=0)

        if dataslice > data.shape[0] - 1:
            dataslice = np.random.randint(0, data.shape[0] - 1)

        start_slice = dataslice
        if dataslice + self.consecutive_slices <= num_slices:
            end_slice = dataslice + self.consecutive_slices
        else:
            end_slice = num_slices

        return data[start_slice:end_slice]

    def process_segmentation_labels(self, segmentation_labels: np.ndarray) -> np.ndarray:
        """
        Process segmentation labels to remove, combine, and separate classes.

        Parameters
        ----------
        segmentation_labels: Segmentation labels to process.
        """
        segmentation_labels = (
            np.moveaxis(segmentation_labels, -1, 0)
            if segmentation_labels.ndim == 3
            else np.moveaxis(segmentation_labels, -1, 1)
        )

        if self.segmentation_classes == 5:
            segmentation_background = segmentation_labels[0]
            segmentation_patellar_cartilage = segmentation_labels[1]
            segmentation_femoral_cartilage = segmentation_labels[2]
            segmentation_lateral_tibial_cartilage = segmentation_labels[3]
            segmentation_medial_tibial_cartilage = segmentation_labels[4]
            segmentation_lateral_meniscus = segmentation_labels[5]
            segmentation_medial_meniscus = segmentation_labels[6]

            segmentation_labels = np.stack(
                [
                    segmentation_background.astype(np.uint8),
                    segmentation_patellar_cartilage.astype(np.uint8),
                    segmentation_femoral_cartilage.astype(np.uint8),
                    (segmentation_lateral_tibial_cartilage + segmentation_medial_tibial_cartilage).astype(np.uint8),
                    (segmentation_lateral_meniscus + segmentation_medial_meniscus).astype(np.uint8),
                ],
                axis=0,
            )
        else:
            segmentation_labels = segmentation_labels.astype(np.uint8)

        # normalize to [0, 1]
        segmentation_labels = segmentation_labels / np.max(segmentation_labels, axis=0, keepdims=True)

        if self.consecutive_slices == 1:
            if segmentation_labels.shape[1] == self.segmentation_classes:
                segmentation_labels = np.moveaxis(segmentation_labels, 1, 0)
            elif segmentation_labels.shape[2] == self.segmentation_classes:
                segmentation_labels = np.moveaxis(segmentation_labels, 2, 0)

        return segmentation_labels

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i: int):
        fname, dataslice, metadata = self.examples[i]
        with h5py.File(fname, "r") as hf:
            kspace = self.get_consecutive_slices(hf, "kspace", dataslice)

            masked_kspace = self.get_consecutive_slices(
                hf, "masked_kspace", dataslice
            ) if "masked_kspace" in hf else None

            if "sensitivity_map" in hf:
                sensitivity_maps = self.get_consecutive_slices(hf, "sensitivity_map", dataslice)
            elif "maps" in hf:
                sensitivity_maps = self.get_consecutive_slices(hf, "maps", dataslice)
            else:
                sensitivity_maps = None

            if "segmentation" in hf:
                segmentation_labels = self.get_consecutive_slices(hf, "segmentation", dataslice)
                segmentation_labels = self.process_segmentation_labels(segmentation_labels)
            else:
                segmentation_labels = None

        # if self.segmentations_root is not None and self.segmentations_root != "None":
        #     # load nii file
        #     seg_fname = os.path.join(self.segmentations_root, fname.name.replace(".h5", ".nii.gz"))
        #     segmentation_labels = np.asarray(nib.load(str(seg_fname)).get_fdata())
        #     # segmentation_labels = np.fft.ifftshift(segmentation_labels, axes=(2))
        #     ky_zero_padding = int((segmentation_labels.shape[1] - 416) / 2)  # hardcoded, see SKM-TEA paper and/or code
        #     kz_zero_padding = int((segmentation_labels.shape[2] - 80) / 2)  # hardcoded, see SKM-TEA paper and/or code
        #     segmentation_labels = segmentation_labels[
        #                           :, ky_zero_padding:-ky_zero_padding, kz_zero_padding:-kz_zero_padding
        #                           ]
        #     # segmentation_labels = np.fft.ifftshift(segmentation_labels, axes=(2))
        #     segmentation_labels = np.transpose(segmentation_labels, (2, 0, 1))
        #     segmentation_labels = self.get_consecutive_slices({"seg": segmentation_labels}, "seg", dataslice)
        #     segmentation_labels = self.process_segmentation_labels(segmentation_labels)
        # else:
        #     segmentation_labels = np.empty([])

        mask = np.empty([])
        imspace = np.empty([])

        return (
            (
                kspace,
                imspace,
                masked_kspace,
                sensitivity_maps,
                mask,
                segmentation_labels,
                fname.name,
                dataslice,
            )
            if self.transform is None
            else self.transform(
                kspace,
                imspace,
                masked_kspace,
                sensitivity_maps,
                mask,
                segmentation_labels,
                fname.name,
                dataslice,
            )
        )
