# coding=utf-8
__author__ = "Dimitrios Karkalousos"

import os
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np

from mridc.collections.common.data.mri_loader import MRIDataset
from mridc.collections.common.parts import utils


class qMRIDataset(MRIDataset):
    """
    A dataset class for loading quantitative MRI data.

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
    n2r_supervised_rate : Optional[float], optional
        A float between 0 and 1. This controls what fraction of the subjects should be loaded for Noise to
        Reconstruction (N2R) supervised loss, if N2R is enabled. Default is ``0.0``.
    transform : Optional[Callable], optional
        A sequence of callable objects that preprocesses the raw data into appropriate form. The transform function
        should take ``kspace``, ``coil sensitivity maps``, ``quantitative maps``, ``mask``, ``initial prediction``,
        ``target``, ``attributes``, ``filename``, and ``slice number`` as inputs. ``target`` may be null for test data.
        Default is ``None``.
    sequence : str, optional
        Sequence of the dataset. For example, ``MEGRE`` or ``FUTURE_SEQUENCES``.
    fixed_precomputed_acceleration : Optional[int], optional
        A list of integers that determine the fixed acceleration of the data. If provided, the data will be loaded with
        the fixed acceleration.
    kspace_scaling_factor : float, optional
        A float that determines the scaling factor of the k-space data. Default is ``10000``.

    Examples
    --------
    >>> from mridc.collections.quantitative.data import qMRIDataset
    >>> dataset = qMRIDataset(root='data/qMRI', sample_rate=0.1, volume_sample_rate=0.1, use_dataset_cache=True)
    >>> print(len(dataset))
    100
    >>> kspace, coil_sensitivities, qmaps, mask, initial_prediction, target, attrs, filename, slice_num = dataset[0]
    >>> print(kspace.shape)
    np.array([30, 640, 368])

    .. note::
        Extends :class:`mridc.collections.common.data.MRIDataset`.
    """

    def __init__(  # noqa: W0221
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
        n2r_supervised_rate: Optional[float] = 0.0,
        transform: Optional[Callable] = None,
        sequence: str = None,
        fixed_precomputed_acceleration: Optional[int] = None,
        kspace_scaling_factor: float = 10000,
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
            n2r_supervised_rate,
            transform,
            **kwargs,
        )
        if sequence not in ("MEGRE", "FUTURE_SEQUENCES"):
            raise ValueError(f'Sequence should be either "MEGRE" or "FUTURE_SEQUENCES". Found {sequence}.')

        self.fixed_precomputed_acceleration = (
            None if utils.is_none(fixed_precomputed_acceleration) else fixed_precomputed_acceleration
        )
        self.kspace_scaling_factor = kspace_scaling_factor

    def check_stored_qdata(self, data: Dict, key: str, dataslice: int) -> List[np.ndarray]:
        """
        Check if quantitative data are stored in the dataset.

        Parameters
        ----------
        data : dict
            Data to extract slices from.
        key : str
            Key to extract slices from. Must be one of ``kspace`` or ``ksp`` or ``reconstruction`` or
            ``sensitivity_map`` or ``sense`` or ``mask`` or ``mask_brain`` or ``mask_head`` or ``B0_map_init`` or
            ``B0_map_target`` or ``S0_map_init`` or ``S0_map_target`` or ``R2star_map_init`` or ``R2star_map_target``
            or ``phi_map_init`` or ``phi_map_target`` or ``prediction`` or ``reconstruction_sense``.
        dataslice : int
            Slice to index.

        Returns
        -------
        List[np.ndarray]
            List of quantitative data.
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

    def __getitem__(self, i: int):  # noqa: W0221
        fname, dataslice, metadata = self.examples[i]
        with h5py.File(fname, "r") as hf:
            kspace = self.get_consecutive_slices(hf, "kspace", dataslice).astype(np.complex64)

            kspace = kspace / self.kspace_scaling_factor

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
            elif any("mask_" in _ for _ in hf.keys()):
                # check for stored masks with acceleration factors
                mask = self.check_stored_qdata(hf, "mask_", dataslice)
            elif self.mask_root is not None and self.mask_root != "None":
                with h5py.File(Path(self.mask_root) / fname.name, "r") as mf:
                    mask = np.asarray(self.get_consecutive_slices(mf, "mask", dataslice))
            else:
                mask = np.empty([])

            mask_brain = (
                np.asarray(self.get_consecutive_slices(hf, "mask_brain", dataslice))
                if "mask_brain" in hf
                else np.empty([])
            )

            mask_head = (
                np.asarray(self.get_consecutive_slices(hf, "mask_head", dataslice))
                if "mask_head" in hf
                else np.empty([])
            )

            mask = [mask, mask_brain, mask_head]

            B0_map_target = (
                self.get_consecutive_slices(hf, "B0_map_target", dataslice) if "B0_map_target" in hf else np.empty([])
            )
            B0_map_init = (
                self.check_stored_qdata(hf, "B0_map_init_", dataslice)
                if any("B0_map_init_" in _ for _ in hf.keys())
                else np.empty([])
            )
            B0_map = [B0_map_init, B0_map_target]

            S0_map_target = (
                self.get_consecutive_slices(hf, "S0_map_target", dataslice) if "S0_map_target" in hf else np.empty([])
            )
            S0_map_init = (
                self.check_stored_qdata(hf, "S0_map_init_", dataslice)
                if any("S0_map_init_" in _ for _ in hf.keys())
                else np.empty([])
            )
            S0_map = [S0_map_init, S0_map_target]

            R2star_map_target = (
                self.get_consecutive_slices(hf, "R2star_map_target", dataslice)
                if "R2star_map_target" in hf
                else np.empty([])
            )
            R2star_map_init = (
                self.check_stored_qdata(hf, "R2star_map_init_", dataslice)
                if any("R2star_map_init_" in _ for _ in hf.keys())
                else np.empty([])
            )
            R2star_map = [R2star_map_init, R2star_map_target]

            phi_map_target = (
                self.get_consecutive_slices(hf, "phi_map_target", dataslice)
                if "phi_map_target" in hf
                else np.empty([])
            )
            phi_map_init = (
                self.check_stored_qdata(hf, "phi_map_init_", dataslice)
                if any("phi_map_init_" in _ for _ in hf.keys())
                else np.empty([])
            )
            phi_map = [phi_map_init, phi_map_target]

            qmaps = [B0_map, S0_map, R2star_map, phi_map]

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

        if self.data_saved_per_slice:
            # arbitrary slice number for logging purposes
            dataslice = str(fname.name)  # type: ignore
            if "h5" in dataslice:  # type: ignore
                dataslice = dataslice.split(".h5")[0]  # type: ignore
            dataslice = int(dataslice.split("_")[-1])  # type: ignore

        return (
            (
                kspace,
                sensitivity_map,
                qmaps,
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
                qmaps,
                mask,
                prediction,
                target,
                attrs,
                fname.name,
                dataslice,
            )
        )
