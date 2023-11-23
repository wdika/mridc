# coding=utf-8
__author__ = "Dimitrios Karkalousos"

import json
import logging
import os
import random
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import yaml  # type: ignore
from defusedxml.ElementTree import fromstring
from torch.utils.data import Dataset

from mridc.collections.common.parts import utils


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


class MRIDataset(Dataset):
    """
    A dataset class for loading (any task) MRI data.

    Parameters
    ----------
    root : Union[str, Path, os.PathLike]
        Path to the dataset.
    coil_sensitivity_maps_root : Union[str, Path, os.PathLike], optional
        Path to the coil sensitivities maps dataset, if stored separately.
    mask_root : Union[str, Path, os.PathLike], optional
        Path to stored masks, if stored separately.
    dataset_format : str, optional
        The format of the dataset. For example, ``'custom_dataset'`` or ``'public_dataset_name'``.
    sample_rate : Optional[float], optional
        A float between 0 and 1. This controls what fraction of the slices should be loaded. When creating subsampled
        datasets either set sample_rates (sample by slices) or volume_sample_rates (sample by volumes) but not both.
    volume_sample_rate : Optional[float], optional
        A float between 0 and 1. This controls what fraction of the volumes should be loaded. When creating subsampled
        datasets either set sample_rates (sample by slices) or volume_sample_rates (sample by volumes) but not both.
    use_dataset_cache : bool, optional
        Whether to cache dataset metadata. This is very useful for large datasets.
    dataset_cache_file : Union[str, Path, os.PathLike, none], optional
        A file in which to cache dataset information for faster load times. If not provided, the cache will be stored
        in the dataset root.
    num_cols : Optional[Tuple[int]], optional
        If provided, only slices with the desired number of columns will be considered.
    consecutive_slices : int, optional
        An int (>0) that determine the amount of consecutive slices of the file to be loaded at the same time.
        Default is ``1``, loading single slices.
    data_saved_per_slice : bool, optional
        Whether the data is saved per slice or per volume.
    n2r_supervised_rate : Optional[float], optional
        A float between 0 and 1. This controls what fraction of the subjects should be loaded for Noise to
        Reconstruction (N2R) supervised loss, if N2R is enabled. Default is ``0.0``.
    transform : Optional[Callable], optional
        A sequence of callable objects that preprocesses the raw data into appropriate form. The transform function
        should take ``kspace``, ``coil sensitivity maps``, ``quantitative maps``, ``mask``, ``initial prediction``,
        ``target``, ``attributes``, ``filename``, and ``slice number`` as inputs. ``target`` may be null for test data.
        Default is ``None``.
    **kwargs
        Additional keyword arguments.

    .. note::
        Extends :class:`torch.utils.data.Dataset`.
    """

    def __init__(  # noqa: C901
        self,  # noqa: C901
        root: Union[str, Path, os.PathLike],
        coil_sensitivity_maps_root: Union[str, Path, os.PathLike] = None,
        mask_root: Union[str, Path, os.PathLike] = None,
        dataset_format: str = None,
        sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
        use_dataset_cache: bool = False,
        dataset_cache_file: Union[str, Path, os.PathLike] = None,
        num_cols: Optional[Tuple[int]] = None,
        consecutive_slices: int = 1,
        data_saved_per_slice: bool = False,
        n2r_supervised_rate: Optional[float] = 0.0,
        transform: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__()
        self.coil_sensitivity_maps_root = coil_sensitivity_maps_root
        self.mask_root = mask_root

        self.dataset_format = dataset_format

        # set default sampling mode if none given
        if not utils.is_none(sample_rate) and not utils.is_none(volume_sample_rate):
            raise ValueError(
                f"Both sample_rate {sample_rate} and volume_sample_rate {volume_sample_rate} are set. "
                "Please set only one of them."
            )

        if sample_rate is None or sample_rate == "None":
            sample_rate = 1.0

        if volume_sample_rate is None or volume_sample_rate == "None":
            volume_sample_rate = 1.0

        self.dataset_cache_file = (
            None if utils.is_none(dataset_cache_file) else Path(dataset_cache_file)  # type: ignore
        )

        if self.dataset_cache_file is not None and self.dataset_cache_file.exists() and use_dataset_cache:
            with open(self.dataset_cache_file, "rb") as f:  # type: ignore
                dataset_cache = yaml.safe_load(f)
        else:
            dataset_cache = {}

        if consecutive_slices < 1:
            raise ValueError(f"Consecutive slices {consecutive_slices} is out of range, must be > 0.")
        self.consecutive_slices = consecutive_slices

        self.transform = transform

        self.data_saved_per_slice = data_saved_per_slice

        self.recons_key = "reconstruction"
        self.examples = []

        # Check if our dataset is in the cache. If yes, use that metadata, if not, then regenerate the metadata.
        if dataset_cache.get(root) is None or not use_dataset_cache:
            if str(root).endswith(".json"):  # type: ignore
                with open(root, "r") as f:  # type: ignore  # noqa: C901
                    examples = json.load(f)
                files = [Path(example) for example in examples]
            else:
                files = list(Path(root).iterdir())

            if n2r_supervised_rate != 0.0:
                # randomly select a subset of files for N2R supervised loss based on n2r_supervised_rate
                n2r_supervised_files = random.sample(
                    files, int(np.round(n2r_supervised_rate * len(files)))  # type: ignore
                )

            for fname in sorted(files):
                metadata, num_slices = self._retrieve_metadata(fname)

                if n2r_supervised_rate != 0.0:
                    metadata["n2r_supervised"] = True if fname in n2r_supervised_files else False
                    logging.info(f"{n2r_supervised_files} files are selected for N2R supervised loss.")
                else:
                    metadata["n2r_supervised"] = False

                if not utils.is_none(num_slices) and not utils.is_none(consecutive_slices):
                    num_slices = num_slices - (consecutive_slices - 1)
                self.examples += [(fname, slice_ind, metadata) for slice_ind in range(num_slices)]

            if dataset_cache.get(root) is None and use_dataset_cache:
                dataset_cache[root] = self.examples
                logging.info("Saving dataset cache to %s.", self.dataset_cache_file)
                with open(self.dataset_cache_file, "wb") as f:  # type: ignore
                    yaml.dump(dataset_cache, f)  # type: ignore
        else:
            logging.info("Using dataset cache from %s.", self.dataset_cache_file)
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

        if num_cols and not utils.is_none(num_cols):
            self.examples = [ex for ex in self.examples if ex[2]["encoding_size"][1] in num_cols]  # type: ignore

    def _retrieve_metadata(self, fname: Union[str, Path]) -> Tuple[Dict, int]:  # noqa: C901
        """
        Retrieve metadata from a given file.

        Parameters
        ----------
        fname : Union[str, Path]
            Path to file.

        Returns
        -------
        Tuple[Dict, int]
            Metadata dictionary and number of slices in the file.

        Examples
        --------
        >>> metadata, num_slices = _retrieve_metadata("file.h5")
        >>> metadata
        {'padding_left': 0, 'padding_right': 0, 'encoding_size': 0, 'recon_size': (0, 0)}
        >>> num_slices
        1
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

                padding_left = enc_size[1] // 2 - enc_limits_center
                padding_right = padding_left + enc_limits_max
            else:
                padding_left = 0
                padding_right = 0
                enc_size = (0, 0, 0)
                recon_size = (0, 0, 0)

            if "kspace" in hf:
                shape = hf["kspace"].shape
            elif "ksp" in hf:
                shape = hf["ksp"].shape
            elif "reconstruction" in hf:
                shape = hf["reconstruction"].shape
            else:
                raise ValueError(f"{fname} does not contain kspace or reconstruction data.")

        num_slices = 1 if self.data_saved_per_slice else shape[0]
        metadata = {
            "padding_left": padding_left,
            "padding_right": padding_right,
            "encoding_size": enc_size,
            "recon_size": recon_size,
        }

        return metadata, num_slices

    def get_consecutive_slices(self, data: Dict, key: str, dataslice: int) -> np.ndarray:
        """
        Get consecutive slices from a given data dictionary.

        Parameters
        ----------
        data : dict
            Data to extract slices from.
        key : str
            Key to extract slices from.
        dataslice : int
            Slice to index.

        Returns
        -------
        np.ndarray
            Array of consecutive slices. If ``self.consecutive_slices`` is > 1, then the array will have shape
            ``(self.consecutive_slices, *data[key].shape[1:])``. Otherwise, the array will have shape
            ``data[key].shape[1:]``.

        Examples
        --------
        >>> data = {"kspace": np.random.rand(10, 640, 368)}
        >>> get_consecutive_slices(data, "kspace", 1).shape
        (1, 640, 368)
        >>> get_consecutive_slices(data, "kspace", 5).shape
        (5, 640, 368)
        """
        x = data[key]

        if self.data_saved_per_slice:
            x = np.expand_dims(x, axis=0)

        if self.consecutive_slices == 1:
            if x.shape[0] == 1:
                return x[0]
            if x.ndim != 2:
                return x[dataslice]
            return x

        num_slices = x.shape[0]
        if self.consecutive_slices > num_slices:
            return np.stack(x, axis=0)

        start_slice = dataslice
        if dataslice + self.consecutive_slices <= num_slices:
            end_slice = dataslice + self.consecutive_slices
        else:
            end_slice = num_slices

        return x[start_slice:end_slice]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i: int):
        raise NotImplementedError
