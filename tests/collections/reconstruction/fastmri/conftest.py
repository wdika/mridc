# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Parts of the code have been taken from https://github.com/facebookresearch/fastMRI

import numpy as np
import pytest
import torch

from tests.collections.reconstruction.fastmri.create_temp_data import create_temp_data

# these are really slow - skip by default
SKIP_INTEGRATIONS = True


def create_input(shape):
    """
    Create a random input tensor of the given shape.

    Args:
        shape: The shape of the input tensor.

    Returns:
        A random input tensor.
    """
    x = np.arange(np.product(shape)).reshape(shape)
    x = torch.from_numpy(x).float()

    return x


@pytest.fixture(scope="session")
def fastmri_mock_dataset(tmp_path_factory):
    """
    Create a mock dataset for testing.

    Args:
        tmp_path_factory: A temporary path factory.

    Returns:
        A mock dataset.
    """
    path = tmp_path_factory.mktemp("fastmri_data")

    return create_temp_data(path)


@pytest.fixture
def skip_integration_tests():
    """
    Skip integration tests if the environment variable is set.

    Returns:
        A boolean indicating whether to skip integration tests.
    """
    return SKIP_INTEGRATIONS


@pytest.fixture
def knee_split_lens():
    """
    The split lengths for the knee dataset.

    Returns:
        A dictionary with the split lengths.
    """
    return {
        "multicoil_train": 34742,
        "multicoil_val": 7135,
        "multicoil_test": 4092,
        "singlecoil_train": 34742,
        "singlecoil_val": 7135,
        "singlecoil_test": 3903,
    }


@pytest.fixture
def brain_split_lens():
    """
    The split lengths for the brain dataset.

    Returns:
        A dictionary with the split lengths.
    """
    return {
        "multicoil_train": 70748,
        "multicoil_val": 21842,
        "multicoil_test": 8852,
    }
