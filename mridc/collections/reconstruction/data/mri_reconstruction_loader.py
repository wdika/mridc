# coding=utf-8
__author__ = "Dimitrios Karkalousos"

# Parts of the code have been taken from https://github.com/facebookresearch/fastMRI

import os
import re
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import h5py
import numpy as np
import yaml  # type: ignore

from mridc.collections.common.data.mri_loader import MRIDataset


class ReconstructionMRIDataset(MRIDataset):
    """
    A dataset class for accelerated-MRI reconstruction.

    Parameters
    ----------
    root : Union[str, Path, os.PathLike]
        Path to the dataset.
    sense_root : Union[str, Path, os.PathLike], optional
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
    dataset_cache_file : Union[str, Path, os.PathLike], optional
        A file in which to cache dataset information for faster load times.
    num_cols : Optional[Tuple[int]], optional
        If provided, only slices with the desired number of columns will be considered.
    consecutive_slices : int, optional
        An int (>0) that determine the amount of consecutive slices of the file to be loaded at the same time.
        Default is ``1``, loading single slices.
    data_saved_per_slice : bool, optional
        Whether the data is saved per slice or per volume.
    transform : Optional[Callable], optional
        A sequence of callable objects that preprocesses the raw data into appropriate form. The transform function
        should take ``kspace``, ``coil sensitivity maps``, ``mask``, ``initial prediction``, ``target``,
        ``attributes``, ``filename``, and ``slice number`` as inputs. ``target`` may be null for test data.
        Default is ``None``.
    **kwargs : dict
        Additional keyword arguments.

    Examples
    --------
    >>> from mridc.collections.reconstruction.data import ReconstructionMRIDataset
    >>> dataset = ReconstructionMRIDataset(root='data/train', sample_rate=0.1)
    >>> print(len(dataset))
    100
    >>> kspace, coil_sensitivities, mask, initial_prediction, target, attrs, filename, slice_num = dataset[0]
    >>> print(kspace.shape)
    np.array([30, 640, 368])

    .. note::
        Extends :class:`mridc.collections.common.data.MRIDataset`.
    """

    def __init__(
        self,
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
        transform: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(
            root,
            coil_sensitivity_maps_root,
            mask_root,
            dataset_format,
            sample_rate,
            volume_sample_rate,
            use_dataset_cache,
            dataset_cache_file,
            num_cols,
            consecutive_slices,
            data_saved_per_slice,
            transform,
            **kwargs,
        )

    def __getitem__(self, i: int):
        fname, dataslice, metadata = self.examples[i]
        with h5py.File(fname, "r") as hf:
            kspace = self.get_consecutive_slices(hf, "kspace", dataslice).astype(np.complex64)

            if self.dataset_format is not None and self.dataset_format.lower() == "cc359":
                kspace = np.transpose(kspace[..., ::2] + 1j * kspace[..., 1::2], (2, 0, 1))

            if "sensitivity_map" in hf:
                sensitivity_map = self.get_consecutive_slices(hf, "sensitivity_map", dataslice).astype(np.complex64)
            elif self.coil_sensitivity_maps_root is not None and self.coil_sensitivity_maps_root != "None":
                with h5py.File(
                    Path(self.coil_sensitivity_maps_root) / Path(str(fname).split("/")[-2]) / fname.name, "r"
                ) as sf:
                    if "sensitivity_map" in sf or "sensitivity_map" in next(iter(sf.keys())):
                        sensitivity_map = (
                            self.get_consecutive_slices(sf, "sensitivity_map", dataslice)
                            .squeeze()
                            .astype(np.complex64)
                        )
            else:
                sensitivity_map = np.array([])

            if "mask" in hf:
                mask = np.asarray(self.get_consecutive_slices(hf, "mask", dataslice))
                if mask.ndim == 3:
                    mask = mask[dataslice]
            elif self.mask_root is not None and self.mask_root != "None":
                if self.dataset_format is not None and self.dataset_format.lower() == "cc359":
                    mask = []
                    with h5py.File(Path(self.mask_root) / fname.name, "r") as mf:
                        for key in mf.keys():
                            mask.append(np.asarray(self.get_consecutive_slices(mf, key, dataslice)))
                else:
                    with h5py.File(Path(self.mask_root) / fname.name, "r") as mf:
                        mask = np.asarray(self.get_consecutive_slices(mf, "mask", dataslice))
            else:
                mask = None

            prediction = (
                self.get_consecutive_slices(hf, "eta", dataslice).astype(np.complex64) if "eta" in hf else np.array([])
            )

            # find key containing "reconstruction_"
            rkey = re.findall(r"reconstruction_(.*)", str(hf.keys()))
            self.recons_key = "reconstruction_" + rkey[0] if rkey else "target"
            if "reconstruction_rss" in self.recons_key:
                self.recons_key = "reconstruction_rss"
            elif "reconstruction_sense" in hf:
                self.recons_key = "reconstruction_sense"

            target = self.get_consecutive_slices(hf, self.recons_key, dataslice) if self.recons_key in hf else None

            attrs = dict(hf.attrs)
            attrs.update(metadata)

        if sensitivity_map.shape != kspace.shape and sensitivity_map.ndim > 1:
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
                prediction,
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
                prediction,
                target,
                attrs,
                fname.name,
                dataslice,
            )
        )
