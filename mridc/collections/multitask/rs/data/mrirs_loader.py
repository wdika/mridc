# coding=utf-8
__author__ = "Dimitrios Karkalousos"

import os
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import h5py
import numpy as np
import yaml  # type: ignore

from mridc.collections.common.data.mri_loader import MRIDataset
from mridc.collections.common.parts.utils import is_none


class RSMRIDataset(MRIDataset):
    """
    A dataset class for accelerated-MRI reconstruction and MRI segmentation.

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
        should take ``kspace``, ``coil sensitivity maps``, ``mask``, ``initial prediction``, ``segmentation``,
        ``target``, ``attributes``, ``filename``, and ``slice number`` as inputs. ``target`` may be null for test data.
        Default is ``None``.
    segmentations_root : Union[str, Path, os.PathLike], optional
        Path to the dataset containing the segmentations.
    initial_predictions_root : Union[str, Path, os.PathLike], optional
        Path to the dataset containing the initial predictions. If provided, the initial predictions will be used as
        the input of the reconstruction network. Default is ``None``.
    segmentation_classes : int, optional
        The number of segmentation classes. Default is ``2``.
    segmentation_classes_to_remove : Optional[Tuple[int]], optional
        A tuple of segmentation classes to remove. For example, if the dataset contains segmentation classes 0, 1, 2,
        3, and 4, and you want to remove classes 1 and 3, set this to ``(1, 3)``. Default is ``None``.
    segmentation_classes_to_combine : Optional[Tuple[int]], optional
        A tuple of segmentation classes to combine. For example, if the dataset contains segmentation classes 0, 1, 2,
        3, and 4, and you want to combine classes 1 and 3, set this to ``(1, 3)``. Default is ``None``.
    segmentation_classes_to_separate : Optional[Tuple[int]], optional
        A tuple of segmentation classes to separate. For example, if the dataset contains segmentation classes 0, 1, 2,
        3, and 4, and you want to separate class 1 into 2 classes, set this to ``(1, 2)``. Default is ``None``.
    segmentation_classes_thresholds : Optional[Tuple[float]], optional
        A tuple of thresholds for the segmentation classes. For example, if the dataset contains segmentation classes
        0, 1, 2, 3, and 4, and you want to set the threshold for class 1 to 0.5, set this to
        ``(0.5, 0.5, 0.5, 0.5, 0.5)``. Default is ``None``.
    complex_data : bool, optional
        Whether the data is complex. If ``False``, the data is assumed to be magnitude only. Default is ``True``.
    **kwargs : dict
        Additional keyword arguments.

    Examples
    --------
    >>> from mridc.collections.multitask.rs.data.mrirs_loader import RSMRIDataset
    >>> dataset = RSMRIDataset(root='data/train', sample_rate=0.1)
    >>> print(len(dataset))
    100
    >>> kspace, imspace, coil_sensitivities, mask, initial_prediction, segmentation_labels, attrs, filename, \
    slice_num = dataset[0]
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
        segmentations_root: Union[str, Path, os.PathLike] = None,
        initial_predictions_root: Union[str, Path, os.PathLike] = None,
        segmentation_classes: int = 2,
        segmentation_classes_to_remove: Optional[Tuple[int]] = None,
        segmentation_classes_to_combine: Optional[Tuple[int]] = None,
        segmentation_classes_to_separate: Optional[Tuple[int]] = None,
        segmentation_classes_thresholds: Optional[Tuple[float]] = None,
        complex_data: bool = True,
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

        self.segmentations_root = segmentations_root
        self.initial_predictions_root = initial_predictions_root

        # Create random number generator used for consecutive slice selection and set consecutive slice amount
        self.consecutive_slices = consecutive_slices
        self.segmentation_classes = segmentation_classes
        self.segmentation_classes_to_remove = segmentation_classes_to_remove
        self.segmentation_classes_to_combine = segmentation_classes_to_combine
        self.segmentation_classes_to_separate = segmentation_classes_to_separate
        self.segmentation_classes_thresholds = segmentation_classes_thresholds
        self.complex_data = complex_data

    def process_segmentation_labels(self, segmentation_labels: np.ndarray) -> np.ndarray:
        """
        Process segmentation labels to remove, combine, and separate classes.

        Parameters
        ----------
        segmentation_labels : np.ndarray
            The segmentation labels. The shape should be (num_slices, height, width) or (height, width).

        Returns
        -------
        np.ndarray
            The processed segmentation labels.
        """
        # make sure that the segmentation dim will be the last one
        segmentation_labels_dim = -1
        for dim in range(segmentation_labels.ndim):
            if segmentation_labels.shape[dim] == self.segmentation_classes:
                segmentation_labels_dim = dim

        if segmentation_labels.ndim == 2:
            segmentation_labels = np.expand_dims(segmentation_labels, axis=0)

        if segmentation_labels.ndim == 3 and segmentation_labels_dim == 0:
            segmentation_labels = np.transpose(segmentation_labels, (1, 2, 0))
        elif segmentation_labels.ndim == 3 and segmentation_labels_dim == 1:
            segmentation_labels = np.transpose(segmentation_labels, (0, 2, 1))
        elif segmentation_labels.ndim == 4 and segmentation_labels_dim == 0:
            segmentation_labels = np.transpose(segmentation_labels, (1, 2, 3, 0))
        elif segmentation_labels.ndim == 4 and segmentation_labels_dim == 1:
            segmentation_labels = np.transpose(segmentation_labels, (0, 2, 3, 1))
        elif segmentation_labels.ndim == 4 and segmentation_labels_dim == 2:
            segmentation_labels = np.transpose(segmentation_labels, (0, 1, 3, 2))

        removed_classes = 0

        # check if we need to remove any classes, e.g. background
        if not is_none(self.segmentation_classes_to_remove):
            segmentation_labels = np.stack(
                [
                    segmentation_labels[..., x]
                    for x in range(segmentation_labels.shape[-1])
                    if x not in self.segmentation_classes_to_remove  # type: ignore
                ],
                axis=-1,
            )
            removed_classes += len(self.segmentation_classes_to_remove)  # type: ignore

        # check if we need to combine any classes, e.g. White Matter and Gray Matter
        if not is_none(self.segmentation_classes_to_combine):
            segmentation_labels_to_combine = []
            segmentation_labels_to_keep = []
            for x in range(segmentation_labels.shape[-1]):
                if x in self.segmentation_classes_to_combine:  # type: ignore
                    segmentation_labels_to_combine.append(segmentation_labels[..., x - removed_classes])
                else:
                    segmentation_labels_to_keep.append(segmentation_labels[..., x - removed_classes])
            segmentation_labels_to_combine = np.expand_dims(
                np.sum(np.stack(segmentation_labels_to_combine, axis=-1), axis=-1), axis=-1
            )
            segmentation_labels_to_keep = np.stack(segmentation_labels_to_keep, axis=-1)

            if self.segmentation_classes_to_remove is not None and (
                0 in self.segmentation_classes_to_remove or "0" in self.segmentation_classes_to_remove
            ):
                # if background is removed, we can stack the combined labels with the rest straight away
                segmentation_labels = np.concatenate(
                    [segmentation_labels_to_combine, segmentation_labels_to_keep], axis=-1
                )
            else:
                # if background is not removed, we need to add it back as new background channel
                segmentation_labels = np.concatenate(
                    [
                        np.expand_dims(segmentation_labels[..., 0], axis=-1),
                        segmentation_labels_to_combine,
                        segmentation_labels_to_keep,
                    ],
                    axis=-1,
                )

            removed_classes += len(self.segmentation_classes_to_combine) - 1  # type: ignore

        # check if we need to separate any classes, e.g. pathologies from White Matter and Gray Matter
        if not is_none(self.segmentation_classes_to_separate):
            for x in self.segmentation_classes_to_separate:  # type: ignore
                segmentation_class_to_separate = segmentation_labels[..., x - removed_classes]
                for i in range(segmentation_labels.shape[-1]):
                    if i == x - removed_classes:
                        continue
                    segmentation_labels[..., i][segmentation_class_to_separate > 0] = 0

        segmentation_labels = (
            np.moveaxis(segmentation_labels, -1, 0)
            if segmentation_labels.ndim == 3
            else np.moveaxis(segmentation_labels, -1, 1)
        )

        # # threshold probability maps if any threshold is given
        if not is_none(self.segmentation_classes_thresholds):
            for i, voxel_thres in enumerate(self.segmentation_classes_thresholds):  # type: ignore
                if not is_none(voxel_thres):
                    segmentation_labels[..., i] = segmentation_labels[..., i] > float(voxel_thres)

        if self.consecutive_slices == 1:
            if segmentation_labels.shape[1] == self.segmentation_classes:
                segmentation_labels = np.moveaxis(segmentation_labels, 1, 0)
            elif segmentation_labels.shape[2] == self.segmentation_classes:
                segmentation_labels = np.moveaxis(segmentation_labels, 2, 0)

        return segmentation_labels

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

                if self.dataset_format is not None and self.dataset_format.lower() == "cc359":
                    kspace = np.transpose(kspace[..., ::2] + 1j * kspace[..., 1::2], (2, 0, 1))

                if "sensitivity_map" in hf:
                    sensitivity_map = self.get_consecutive_slices(hf, "sensitivity_map", dataslice).astype(
                        np.complex64
                    )
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
                                tmp = np.asarray(self.get_consecutive_slices(mf, key, 1))
                                if self.consecutive_slices > 1:
                                    tmp = np.repeat(tmp, self.consecutive_slices, axis=0)
                                mask.append(tmp)
                    else:
                        with h5py.File(Path(self.mask_root) / fname.name, "r") as mf:
                            mask = np.asarray(self.get_consecutive_slices(mf, "mask", dataslice))
                else:
                    mask = None
                imspace = np.empty([])

            elif not self.complex_data:
                if "reconstruction_rss" in hf:
                    imspace = self.get_consecutive_slices(hf, "reconstruction_rss", dataslice)
                elif "reconstruction_sense" in hf:
                    imspace = self.get_consecutive_slices(hf, "reconstruction_sense", dataslice)
                elif "reconstruction" in hf:
                    imspace = self.get_consecutive_slices(hf, "reconstruction", dataslice)
                elif "target" in hf:
                    imspace = self.get_consecutive_slices(hf, "target", dataslice)
                else:
                    raise ValueError(
                        "Complex data has not been selected but no reconstruction data found in file. "
                        "Only 'reconstruction' key is supported."
                    )
                kspace = np.empty([])
                sensitivity_map = np.array([])
                mask = np.empty([])

            if self.segmentations_root is not None and self.segmentations_root != "None":
                with h5py.File(Path(self.segmentations_root) / fname.name, "r") as sf:
                    segmentation_labels = np.asarray(self.get_consecutive_slices(sf, "segmentation", dataslice))
                    segmentation_labels = self.process_segmentation_labels(segmentation_labels)
            elif "segmentation" in hf:
                segmentation_labels = np.asarray(self.get_consecutive_slices(hf, "segmentation", dataslice))
                segmentation_labels = self.process_segmentation_labels(segmentation_labels)
            else:
                segmentation_labels = np.empty([])

            if not is_none(self.initial_predictions_root):
                with h5py.File(Path(self.initial_predictions_root) / fname.name, "r") as pf:  # type: ignore
                    initial_prediction = self.get_consecutive_slices(pf, "reconstruction", dataslice)
                    initial_prediction = initial_prediction.squeeze().astype(np.complex64)
            else:
                initial_prediction = np.empty([])

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
                imspace,
                sensitivity_map,
                mask,
                initial_prediction,
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
                initial_prediction,
                segmentation_labels,
                attrs,
                fname.name,
                dataslice,
            )
        )
