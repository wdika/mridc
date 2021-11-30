# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

import numpy as np
import pytest

from mridc.collections.reconstruction.data.subsample import (
    EquispacedMaskFunc,
    Gaussian1DMaskFunc,
    Gaussian2DMaskFunc,
    Poisson2DMaskFunc,
    RandomMaskFunc,
    create_mask_for_mask_type,
)


@pytest.mark.parametrize(
    "mask_type, center_fractions, accelerations, expected_mask_func, x, seed, half_scan_percentage",
    [("random1d", [0.08, 0.04], [4, 8], RandomMaskFunc, np.array([1, 320, 320]), None, 0)],
)
def test_create_mask_for_random_type(
    mask_type, center_fractions, accelerations, expected_mask_func, x, seed, half_scan_percentage
):
    """
    Test that the function returns random 1D masks

    Args:
        mask_type: The type of mask to be created
        center_fractions: The center fractions of the mask
        accelerations: The accelerations of the mask
        expected_mask_func: The expected mask function
        x: The shape of the mask
        seed: The seed of the mask
        half_scan_percentage: The half scan percentage of the mask

    Returns:
        None
    """
    mask_func = create_mask_for_mask_type(mask_type, center_fractions, accelerations)

    mask, acc = mask_func(x, seed, half_scan_percentage)
    mask = mask.squeeze(0).numpy()
    x[-2] = 1

    if not isinstance(mask_func, expected_mask_func):
        raise AssertionError
    if not accelerations[0] <= mask_func.choose_acceleration()[1] <= accelerations[1]:
        raise AssertionError
    if mask.shape != (*x[2:], 1):
        raise AssertionError
    if mask.dtype != np.float32:
        raise AssertionError
    if not accelerations[0] <= acc <= accelerations[1]:
        raise AssertionError


@pytest.mark.parametrize(
    "mask_type, center_fractions, accelerations, expected_mask_func, x, seed, half_scan_percentage",
    [("equispaced1d", [0.08, 0.04], [4, 8], EquispacedMaskFunc, np.array([1, 320, 320]), None, 0)],
)
def test_create_mask_for_equispaced_type(
    mask_type, center_fractions, accelerations, expected_mask_func, x, seed, half_scan_percentage
):
    """
    Test that the function returns equispaced 1D masks

    Args:
        mask_type: The type of mask to be created
        center_fractions: The center fractions of the mask
        accelerations: The accelerations of the mask
        expected_mask_func: The expected mask function
        x: The shape of the mask
        seed: The seed of the mask
        half_scan_percentage: The half scan percentage of the mask

    Returns:
        None
    """
    mask_func = create_mask_for_mask_type(mask_type, center_fractions, accelerations)

    mask, acc = mask_func(x, seed, half_scan_percentage)
    mask = mask.squeeze(0).numpy()
    x[-2] = 1

    if not isinstance(mask_func, expected_mask_func):
        raise AssertionError
    if not accelerations[0] <= mask_func.choose_acceleration()[1] <= accelerations[1]:
        raise AssertionError
    if mask.shape != (*x[2:], 1):
        raise AssertionError
    if mask.dtype != np.float32:
        raise AssertionError
    if not accelerations[0] <= acc <= accelerations[1]:
        raise AssertionError


@pytest.mark.parametrize(
    "mask_type, center_fractions, accelerations, expected_mask_func, x, seed, half_scan_percentage, scale",
    [("gaussian1d", [0.7, 0.7], [4, 10], Gaussian1DMaskFunc, np.array([1, 320, 320, 1]), None, 0, 0.02)],
)
def test_create_mask_for_gaussian1d_type(
    mask_type, center_fractions, accelerations, expected_mask_func, x, seed, half_scan_percentage, scale
):
    """
    Test that the function returns gaussian 1D masks

    Args:
        mask_type: The type of mask to be created
        center_fractions: The center fractions of the mask
        accelerations: The accelerations of the mask
        expected_mask_func: The expected mask function
        x: The shape of the mask
        seed: The seed of the mask
        half_scan_percentage: The half scan percentage of the mask
        scale: The scale of the mask

    Returns:
        None
    """
    mask_func = create_mask_for_mask_type(mask_type, center_fractions, accelerations)

    mask, acc = mask_func(x, seed, half_scan_percentage, scale)
    mask = mask.squeeze(0).numpy()
    x[-3] = 1

    if not isinstance(mask_func, expected_mask_func):
        raise AssertionError
    if not accelerations[0] <= mask_func.choose_acceleration()[1] <= accelerations[1]:
        raise AssertionError
    if mask.shape != tuple(x[1:]):
        raise AssertionError
    if mask.dtype != np.float32:
        raise AssertionError
    if not accelerations[0] <= acc <= accelerations[1]:
        raise AssertionError


@pytest.mark.parametrize(
    "mask_type, center_fractions, accelerations, expected_mask_func, x, seed, half_scan_percentage, scale",
    [("gaussian2d", [0.7, 0.7], [4, 10], Gaussian2DMaskFunc, np.array([1, 320, 320, 1]), None, 0, 0.02)],
)
def test_create_mask_for_gaussian2d_type(
    mask_type, center_fractions, accelerations, expected_mask_func, x, seed, half_scan_percentage, scale
):
    """
    Test that the function returns gaussian 2D masks

    Args:
        mask_type: The type of mask to be created
        center_fractions: The center fractions of the mask
        accelerations: The accelerations of the mask
        expected_mask_func: The expected mask function
        x: The shape of the mask
        seed: The seed of the mask
        half_scan_percentage: The half scan percentage of the mask
        scale: The scale of the mask

    Returns:
        None
    """
    mask_func = create_mask_for_mask_type(mask_type, center_fractions, accelerations)

    mask, acc = mask_func(x, seed, half_scan_percentage, scale)
    mask = mask.squeeze(0).squeeze(-1).numpy()

    if not isinstance(mask_func, expected_mask_func):
        raise AssertionError
    if not accelerations[0] <= mask_func.choose_acceleration()[1] <= accelerations[1]:
        raise AssertionError
    if mask.shape != tuple(x[1:-1]):
        raise AssertionError
    if mask.dtype != np.float32:
        raise AssertionError
    if not accelerations[0] <= acc <= accelerations[1]:
        raise AssertionError


@pytest.mark.parametrize(
    "mask_type, center_fractions, accelerations, expected_mask_func, x, seed, half_scan_percentage, scale",
    [("poisson2d", [0.7, 0.7], [4, 10], Poisson2DMaskFunc, np.array([1, 320, 320, 1]), None, 0, 0.02)],
)
def test_create_mask_for_poisson2d_type(
    mask_type, center_fractions, accelerations, expected_mask_func, x, seed, half_scan_percentage, scale
):
    """
    Test that the function returns poisson 2D masks

    Args:
        mask_type: The type of mask to be created
        center_fractions: The center fractions of the mask
        accelerations: The accelerations of the mask
        expected_mask_func: The expected mask function
        x: The shape of the mask
        seed: The seed of the mask
        half_scan_percentage: The half scan percentage of the mask
        scale: The scale of the mask

    Returns:
        None
    """
    mask_func = create_mask_for_mask_type(mask_type, center_fractions, accelerations)

    mask, acc = mask_func(x, seed, half_scan_percentage, scale)
    mask = mask.squeeze(0).squeeze(-1).numpy()

    if not isinstance(mask_func, expected_mask_func):
        raise AssertionError
    if not accelerations[0] <= mask_func.choose_acceleration()[1] <= accelerations[1]:
        raise AssertionError
    if mask.shape != tuple(x[1:-1]):
        raise AssertionError
    if mask.dtype != np.float32:
        raise AssertionError
    if not accelerations[0] <= acc <= accelerations[1]:
        raise AssertionError
