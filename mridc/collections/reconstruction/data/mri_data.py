# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Parts of the code have been taken from https://github.com/facebookresearch/fastMRI

import logging
import os
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import torch
import yaml  # type: ignore
from defusedxml.ElementTree import fromstring
from torch.utils.data import Dataset

from mridc.collections.common.parts.utils import is_none


def et_query(root: str, qlist: Sequence[str], namespace: str = "https://www.ismrm.org/ISMRMRD") -> str:
    """
    Query an XML element for a list of attributes.

    Parameters
    ----------
    root: The root element of the XML tree.
    qlist: A list of strings, each of which is an attribute name.
    namespace: The namespace of the XML tree.

    Returns
    -------
    A string containing the value of the last attribute in the list.
    """
    s = "."
    prefix = "ismrmrd_namespace"

    ns = {prefix: namespace}

    for el in qlist:
        s += f"//{prefix}:{el}"

    value = root.find(s, ns)  # type: ignore
    if value is None:
        return "0"
    return str(value.text)  # type: ignore


class FastMRICombinedSliceDataset(torch.utils.data.Dataset):
    """A dataset that combines multiple datasets."""

    def __init__(
        self,
        roots: Sequence[Path],
        challenges: Sequence[str],
        sense_roots: Optional[Sequence[Path]] = None,
        transforms: Optional[Sequence[Optional[Callable]]] = None,
        sample_rates: Optional[Sequence[Optional[float]]] = None,
        volume_sample_rates: Optional[Sequence[Optional[float]]] = None,
        use_dataset_cache: bool = False,
        dataset_cache_file: Union[str, Path, os.PathLike] = "dataset_cache.yaml",
        num_cols: Optional[Tuple[int]] = None,
    ):
        """
        Parameters
        ----------
        roots: Paths to the datasets.
        challenges: "singlecoil" or "multicoil" depending on which challenge to use.
        sense_roots: Load pre-computed (stored) sensitivity maps.
        transforms: Optional; A sequence of callable objects that preprocesses the raw data into appropriate form.
            The transform function should take 'kspace', 'target', 'attributes', 'filename', and 'slice' as inputs.
            'target' may be null for test data.
        sample_rates: Optional; A sequence of floats between 0 and 1. This controls what fraction of the slices
            should be loaded. When creating subsampled datasets either set sample_rates (sample by slices) or
            volume_sample_rates (sample by volumes) but not both.
        volume_sample_rates: Optional; A sequence of floats between 0 and 1. This controls what fraction of the
            volumes should be loaded. When creating subsampled datasets either set sample_rates (sample by slices)
            or volume_sample_rates (sample by volumes) but not both.
        use_dataset_cache: Whether to cache dataset metadata. This is very useful for large datasets like the brain
            data.
        dataset_cache_file: Optional; A file in which to cache dataset information for faster load times.
        num_cols: Optional; If provided, only slices with the desired number of columns will be considered.
        """
        if sample_rates is not None and volume_sample_rates is not None:
            raise ValueError(
                "either set sample_rates (sample by slices) or volume_sample_rates (sample by volumes) but not both"
            )
        if transforms is None:
            transforms = [None] * len(roots)
        if sample_rates is None:
            sample_rates = [None] * len(roots)
        if volume_sample_rates is None:
            volume_sample_rates = [None] * len(roots)
        if not len(roots) == len(transforms) == len(challenges) == len(sample_rates) == len(volume_sample_rates):
            raise ValueError("Lengths of roots, transforms, challenges, sample_rates do not match")

        self.datasets = []
        self.examples: List[Tuple[Path, int, Dict[str, object]]] = []
        for i, _ in enumerate(roots):
            self.datasets.append(
                FastMRISliceDataset(
                    root=roots[i],
                    transform=transforms[i],
                    sense_root=sense_roots[i] if sense_roots is not None else None,
                    challenge=challenges[i],
                    sample_rate=sample_rates[i],
                    volume_sample_rate=volume_sample_rates[i],
                    use_dataset_cache=use_dataset_cache,
                    dataset_cache_file=dataset_cache_file,
                    num_cols=num_cols,
                )
            )

            self.examples += self.datasets[-1].examples

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    def __getitem__(self, i):
        for dataset in self.datasets:
            if i < len(dataset):
                return dataset[i]
            i = i - len(dataset)


class FastMRISliceDataset(Dataset):
    """A dataset that loads slices from a single dataset."""

    def __init__(
        self,
        root: Union[str, Path, os.PathLike],
        challenge: str = "segmentation",
        transform: Optional[Callable] = None,
        sense_root: Union[str, Path, os.PathLike] = None,
        use_dataset_cache: bool = False,
        sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
        dataset_cache_file: Union[str, Path, os.PathLike] = "dataset_cache.yaml",
        num_cols: Optional[Tuple[int]] = None,
        mask_root: Union[str, Path, os.PathLike] = None,
        consecutive_slices: int = 1,
    ):
        """
        Parameters
        ----------
        root: Path to the dataset.
        challenge: "singlecoil" or "multicoil" depending on which challenge to use.
        transform: Optional; A sequence of callable objects that preprocesses the raw data into appropriate form.
            The transform function should take 'kspace', 'target', 'attributes', 'filename', and 'slice' as inputs.
            'target' may be null for test data.
        sense_root: Path to the coil sensitivities maps dataset.
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
        """
        if challenge not in ("singlecoil", "multicoil", "segmentation"):
            raise ValueError('challenge should be either "singlecoil" or "multicoil" or "segmentation"')
        self.challenge = challenge

        if sample_rate is not None and volume_sample_rate is not None:
            raise ValueError(
                "either set sample_rate (sample by slices) or volume_sample_rate (sample by volumes) but not both"
            )

        self.sense_root = sense_root
        self.mask_root = mask_root

        self.dataset_cache_file = Path(dataset_cache_file)

        self.transform = transform
        self.recons_key = "reconstruction_esc" if challenge == "singlecoil" else "reconstruction_rss"
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
                metadata, num_slices = self._retrieve_metadata(fname)
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
    def _retrieve_metadata(fname):
        """
        Retrieve metadata from a given file.

        Parameters
        ----------
        fname: Path to file.

        Returns
        -------
        A dictionary containing the metadata.
        """
        with h5py.File(fname, "r") as hf:
            if "ismrmrd_header" in hf:
                et_root = fromstring(hf["ismrmrd_header"][()])

                enc = ["encoding", "encodedSpace", "matrixSize"]
                enc_size = (
                    int(et_query(et_root, enc + ["x"])),
                    int(et_query(et_root, enc + ["y"])),
                    int(et_query(et_root, enc + ["z"])),
                )
                rec = ["encoding", "reconSpace", "matrixSize"]
                recon_size = (
                    int(et_query(et_root, rec + ["x"])),
                    int(et_query(et_root, rec + ["y"])),
                    int(et_query(et_root, rec + ["z"])),
                )

                params = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
                enc_limits_center = int(et_query(et_root, params + ["center"]))
                enc_limits_max = int(et_query(et_root, params + ["maximum"])) + 1

                padding_left = torch.div(enc_size[1], 2, rounding_mode="trunc").item() - enc_limits_center
                padding_right = padding_left + enc_limits_max
            else:
                padding_left = 0
                padding_right = 0
                enc_size = 0
                recon_size = (0, 0)

            num_slices = hf["kspace"].shape[0] if "kspace" in hf else hf["reconstruction"].shape[0]

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

        if self.consecutive_slices == 1:
            return data[dataslice] if data.ndim != 2 else data  # mask is 2D

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
        if self.challenge == "segmentation":
            with h5py.File(fname, "r") as hf:
                data = self.get_consecutive_slices(hf, "reconstruction", dataslice)  # .astype(np.complex64)
                if data.ndim == 2:
                    data = np.expand_dims(data, 0)
                # phase = np.tile(
                #     np.linspace(-np.pi, np.pi, int(data.shape[-2] / 2))[:, None], (data.shape[-1], 2, data.shape[-3])
                # ).T
                # data = data * np.exp(1j*phase)
                # data = np.stack([data.real, data.imag], 1)
                # data = np.stack([data, data], 1)
                # data = data / np.max(np.abs(data))
                return (
                    data,
                    np.zeros_like(data),
                    fname.name,
                    dataslice,
                )
        else:
            with h5py.File(fname, "r") as hf:
                kspace = self.get_consecutive_slices(hf, "kspace", dataslice).astype(np.complex64)

                if "sensitivity_map" in hf:
                    sensitivity_map = self.get_consecutive_slices(hf, "sensitivity_map", dataslice).astype(
                        np.complex64
                    )
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
                    mask = [np.load(str(m)) for m in list(Path(self.mask_root).iterdir())]
                else:
                    mask = None

                eta = (
                    self.get_consecutive_slices(hf, "eta", dataslice).astype(np.complex64)
                    if "eta" in hf
                    else np.array([])
                )

                if "reconstruction_sense" in hf:
                    self.recons_key = "reconstruction_sense"

                target = (
                    self.get_consecutive_slices(hf, self.recons_key, dataslice).astype(np.float32)
                    if self.recons_key in hf
                    else None
                )

                attrs = dict(hf.attrs)
                attrs |= metadata

            if sensitivity_map.shape != kspace.shape:
                if sensitivity_map.ndim == 3:
                    sensitivity_map = np.transpose(sensitivity_map, (2, 0, 1))
                elif sensitivity_map.ndim == 4:
                    sensitivity_map = np.transpose(sensitivity_map, (0, 3, 1, 2))
                else:
                    raise ValueError(
                        f"Sensitivity map has invalid dimensions {sensitivity_map.shape} compared to kspace {kspace.shape}"
                    )

            return (
                (
                    kspace,
                    sensitivity_map,
                    mask,
                    eta,
                    target,
                    attrs,
                    fname.name,
                    dataslice,
                )
                if self.transform is None
                else self.transform(
                    kspace,
                    sensitivity_map,
                    mask,
                    eta,
                    target,
                    attrs,
                    fname.name,
                    dataslice,
                )
            )
