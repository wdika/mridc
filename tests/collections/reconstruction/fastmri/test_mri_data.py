# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Parts of the code have been taken from https://github.com/facebookresearch/fastMRI
from mridc.collections.reconstruction.data.mri_data import FastMRICombinedSliceDataset, FastMRISliceDataset


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

    monkeypatch.setattr(FastMRISliceDataset, "_retrieve_metadata", retrieve_metadata_mock)

    for challenge in ("multicoil", "singlecoil"):
        for split in ("train", "val", "test", "challenge"):
            dataset = FastMRISliceDataset(knee_path / f"{challenge}_{split}", transform=None, challenge=challenge)

            if len(dataset) <= 0:
                raise AssertionError
            if dataset is None:
                raise AssertionError

    for challenge in ("multicoil",):
        for split in ("train", "val", "test", "challenge"):
            dataset = FastMRISliceDataset(brain_path / f"{challenge}_{split}", transform=None, challenge=challenge)

            if len(dataset) <= 0:
                raise AssertionError
            if dataset is None:
                raise AssertionError


def test_combined_slice_dataset(fastmri_mock_dataset, monkeypatch):
    """
    Test the combined slice datasets

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

    monkeypatch.setattr(FastMRISliceDataset, "_retrieve_metadata", retrieve_metadata_mock)

    roots = [knee_path / "multicoil_train", knee_path / "multicoil_val"]
    challenges = ["multicoil", "multicoil"]
    transforms = [None, None]

    dataset1 = FastMRISliceDataset(root=roots[0], challenge=challenges[0], transform=transforms[0])
    dataset2 = FastMRISliceDataset(root=roots[1], challenge=challenges[1], transform=transforms[1])
    comb_dataset = FastMRICombinedSliceDataset(roots=roots, challenges=challenges, transforms=transforms)

    if len(comb_dataset) != len(dataset1) + len(dataset2):
        raise AssertionError
    if comb_dataset is None:
        raise AssertionError

    roots = [brain_path / "multicoil_train", brain_path / "multicoil_val"]
    challenges = ["multicoil", "multicoil"]
    transforms = [None, None]

    dataset1 = FastMRISliceDataset(root=roots[0], challenge=challenges[0], transform=transforms[0])
    dataset2 = FastMRISliceDataset(root=roots[1], challenge=challenges[1], transform=transforms[1])
    comb_dataset = FastMRICombinedSliceDataset(roots=roots, challenges=challenges, transforms=transforms)

    if len(comb_dataset) != len(dataset1) + len(dataset2):
        raise AssertionError
    if comb_dataset is None:
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

    monkeypatch.setattr(FastMRISliceDataset, "_retrieve_metadata", retrieve_metadata_mock)

    for challenge in ("multicoil", "singlecoil"):
        for split in ("train", "val", "test", "challenge"):
            dataset = FastMRISliceDataset(knee_path / f"{challenge}_{split}", transform=None, challenge=challenge)

            if len(dataset) <= 0:
                raise AssertionError
            if dataset is None:
                raise AssertionError

    for challenge in ("multicoil",):
        for split in ("train", "val", "test", "challenge"):
            dataset = FastMRISliceDataset(brain_path / f"{challenge}_{split}", transform=None, challenge=challenge)

            if len(dataset) <= 0:
                raise AssertionError
            if dataset is None:
                raise AssertionError


def test_combined_slice_dataset_with_transform(fastmri_mock_dataset, monkeypatch):
    """
    Test the combined slice datasets with transforms

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

    monkeypatch.setattr(FastMRISliceDataset, "_retrieve_metadata", retrieve_metadata_mock)

    roots = [knee_path / "multicoil_train", knee_path / "multicoil_val"]
    challenges = ["multicoil", "multicoil"]
    transforms = [None, None]

    dataset1 = FastMRISliceDataset(root=roots[0], challenge=challenges[0], transform=transforms[0])
    dataset2 = FastMRISliceDataset(root=roots[1], challenge=challenges[1], transform=transforms[1])
    comb_dataset = FastMRICombinedSliceDataset(roots=roots, challenges=challenges, transforms=transforms)

    if len(comb_dataset) != len(dataset1) + len(dataset2):
        raise AssertionError
    if comb_dataset is None:
        raise AssertionError

    roots = [brain_path / "multicoil_train", brain_path / "multicoil_val"]
    challenges = ["multicoil", "multicoil"]
    transforms = [None, None]

    dataset1 = FastMRISliceDataset(root=roots[0], challenge=challenges[0], transform=transforms[0])
    dataset2 = FastMRISliceDataset(root=roots[1], challenge=challenges[1], transform=transforms[1])
    comb_dataset = FastMRICombinedSliceDataset(roots=roots, challenges=challenges, transforms=transforms)

    if len(comb_dataset) != len(dataset1) + len(dataset2):
        raise AssertionError
    if comb_dataset is None:
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

    monkeypatch.setattr(FastMRISliceDataset, "_retrieve_metadata", retrieve_metadata_mock)

    for split in ("train", "val", "test", "challenge"):
        dataset = FastMRISliceDataset(knee_path / f"multicoil_{split}", transform=None, challenge="multicoil")

        if len(dataset) <= 0:
            raise AssertionError
        if dataset is None:
            raise AssertionError

    for split in ("train", "val", "test", "challenge"):
        dataset = FastMRISliceDataset(brain_path / f"multicoil_{split}", transform=None, challenge="multicoil")

        if len(dataset) <= 0:
            raise AssertionError
        if dataset is None:
            raise AssertionError


def test_combined_slice_dataset_with_transform_and_challenge(fastmri_mock_dataset, monkeypatch):
    """
    Test the combined slice datasets with transforms and challenge

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

    monkeypatch.setattr(FastMRISliceDataset, "_retrieve_metadata", retrieve_metadata_mock)

    roots = [knee_path / "multicoil_train", knee_path / "multicoil_val"]
    challenges = ["multicoil", "multicoil"]
    transforms = [None, None]

    dataset1 = FastMRISliceDataset(root=roots[0], challenge=challenges[0], transform=transforms[0])
    dataset2 = FastMRISliceDataset(root=roots[1], challenge=challenges[1], transform=transforms[1])
    comb_dataset = FastMRICombinedSliceDataset(roots=roots, challenges=challenges, transforms=transforms)

    if len(comb_dataset) != len(dataset1) + len(dataset2):
        raise AssertionError
    if comb_dataset is None:
        raise AssertionError

    roots = [brain_path / "multicoil_train", brain_path / "multicoil_val"]
    challenges = ["multicoil", "multicoil"]
    transforms = [None, None]

    dataset1 = FastMRISliceDataset(root=roots[0], challenge=challenges[0], transform=transforms[0])
    dataset2 = FastMRISliceDataset(root=roots[1], challenge=challenges[1], transform=transforms[1])
    comb_dataset = FastMRICombinedSliceDataset(roots=roots, challenges=challenges, transforms=transforms)

    if len(comb_dataset) != len(dataset1) + len(dataset2):
        raise AssertionError
    if comb_dataset is None:
        raise AssertionError
