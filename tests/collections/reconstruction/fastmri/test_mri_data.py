# coding=utf-8
__author__ = "Dimitrios Karkalousos"

# Parts of the code have been taken from https://github.com/facebookresearch/fastMRI
from mridc.collections.reconstruction.data.mri_data import MRISliceDataset


def test_slice_datasets(fastmri_mock_dataset, monkeypatch):
    """
    Test the slice datasets

    Args:
        fastmri_mock_dataset: fastMRI mock dataset
        monkeypatch: monkeypatch

    Returns:
        None
    """
    knee_path, brain_path, metadata = fastmri_mock_dataset

    def retrieve_metadata_mock(_, fname):
        """
        Mock the metadata retrieval

        Args:
            _: ignored
            fname: filename

        Returns:
            metadata: metadata
        """
        return metadata[str(fname)]

    monkeypatch.setattr(
        MRISliceDataset, "_retrieve_metadata", retrieve_metadata_mock)

    for challenge in ("multicoil", "singlecoil"):
        for split in ("train", "val", "test", "challenge"):
            dataset = MRISliceDataset(
                knee_path / f"{challenge}_{split}", transform=None, challenge=challenge)

            if len(dataset) <= 0:
                raise AssertionError
            if dataset is None:
                raise AssertionError

    for challenge in ("multicoil",):
        for split in ("train", "val", "test", "challenge"):
            dataset = MRISliceDataset(
                brain_path / f"{challenge}_{split}", transform=None, challenge=challenge)

            if len(dataset) <= 0:
                raise AssertionError
            if dataset is None:
                raise AssertionError


def test_slice_dataset_with_transform(fastmri_mock_dataset, monkeypatch):
    """
    Test the slice datasets with transforms

    Args:
        fastmri_mock_dataset: fastMRI mock dataset
        monkeypatch: monkeypatch

    Returns:
        None
    """
    knee_path, brain_path, metadata = fastmri_mock_dataset

    def retrieve_metadata_mock(_, fname):
        """
        Mock the metadata retrieval

        Args:
            _: ignored
            fname: filename

        Returns:
            metadata: metadata
        """
        return metadata[str(fname)]

    monkeypatch.setattr(
        MRISliceDataset, "_retrieve_metadata", retrieve_metadata_mock)

    for challenge in ("multicoil", "singlecoil"):
        for split in ("train", "val", "test", "challenge"):
            dataset = MRISliceDataset(
                knee_path / f"{challenge}_{split}", transform=None, challenge=challenge)

            if len(dataset) <= 0:
                raise AssertionError
            if dataset is None:
                raise AssertionError

    for challenge in ("multicoil",):
        for split in ("train", "val", "test", "challenge"):
            dataset = MRISliceDataset(
                brain_path / f"{challenge}_{split}", transform=None, challenge=challenge)

            if len(dataset) <= 0:
                raise AssertionError
            if dataset is None:
                raise AssertionError


def test_slice_dataset_with_transform_and_challenge(fastmri_mock_dataset, monkeypatch):
    """
    Test the slice datasets with transforms and challenge

    Args:
        fastmri_mock_dataset: fastMRI mock dataset
        monkeypatch: monkeypatch

    Returns:
        None
    """
    knee_path, brain_path, metadata = fastmri_mock_dataset

    def retrieve_metadata_mock(_, fname):
        """
        Mock the metadata retrieval

        Args:
            _: ignored
            fname: filename

        Returns:
            metadata: metadata
        """
        return metadata[str(fname)]

    monkeypatch.setattr(
        MRISliceDataset, "_retrieve_metadata", retrieve_metadata_mock)

    for split in ("train", "val", "test", "challenge"):
        dataset = MRISliceDataset(
            knee_path / f"multicoil_{split}", transform=None, challenge="multicoil")

        if len(dataset) <= 0:
            raise AssertionError
        if dataset is None:
            raise AssertionError

    for split in ("train", "val", "test", "challenge"):
        dataset = MRISliceDataset(
            brain_path / f"multicoil_{split}", transform=None, challenge="multicoil")

        if len(dataset) <= 0:
            raise AssertionError
        if dataset is None:
            raise AssertionError
