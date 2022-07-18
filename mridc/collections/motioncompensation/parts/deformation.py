# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from https://github.com/wustl-cig/DeCoLearn/blob/main/decolearn/dataset/modl.py

import warnings
from typing import Sequence, Tuple

import numpy as np
import torch
from skimage.filters import gaussian
from torch.nn import functional as F

warnings.filterwarnings("ignore")


def generate_affine_grid(
    shape: Sequence[int],
    translation: Sequence[float] = (0, 0),
    reflection: Sequence[float] = (1, 1),
    scale: float = 1,
    rotate: int = 0,
    shear: Sequence[int] = (0, 0),
) -> torch.Tensor:
    """
    Generate a grid of points for affine transformation.

    Parameters
    ----------
    shape : Sequence[int]
        Shape of the grid.
    translation : Tuple[float, float]
        Translation of the grid.
    reflection : Tuple[float, float]
        Reflection of the grid.
    scale : float
        Scale of the grid.
    rotate : float
        Rotation of the grid.
    shear : Tuple[float, float]
        Shear of the grid.
    """
    T_translation = np.array([[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]], dtype=object).astype(
        np.float32
    )

    T_reflection = np.array([[reflection[0], 0, 0], [0, reflection[1], 0], [0, 0, 1]], dtype=object).astype(np.float32)

    T_scale = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=object).astype(np.float32)

    rotate = rotate / 180 * np.pi
    T_rotate = np.array(
        [[np.cos(rotate), -np.sin(rotate), 0], [np.sin(rotate), np.cos(rotate), 0], [0, 0, 1]], dtype=object
    ).astype(np.float32)

    T_shear = np.array([[1, shear[0], 0], [shear[1], 1, 0], [0, 0, 1]], dtype=object).astype(np.float32)

    rec = np.matmul(np.matmul(np.matmul(np.matmul(T_translation, T_reflection), T_scale), T_rotate), T_shear)
    rec = rec[:2, :]
    rec = torch.from_numpy(rec)
    theta = rec.unsqueeze_(0)

    return F.affine_grid(theta=theta, size=(1,) + shape)  # type: ignore


def generate_nonlinear_grid(
    shape: Sequence[int],
    P: float,
    theta: float,
    sigma: float,
):
    """
    Generate a grid of points for nonlinear transformation.

    Parameters
    ----------
    shape : Sequence[int]
        Shape of the grid.
    P : float
        Order of the polynomial.
    theta : float
        Angle of the polynomial.
    sigma : float
        Sigma of the polynomial.
    """
    grid = generate_affine_grid(shape=shape).numpy()

    P = int(P)

    if P > 0:
        P_index = np.stack(np.meshgrid(range(shape[-2]), range(shape[-1])), -1)
        P_index = P_index.reshape([-1, 2])

        P_index_choice = np.arange(P_index.shape[0])
        np.random.shuffle(P_index_choice)

        P_index = P_index[P_index_choice[:P]]

        P_index_matrix = np.zeros([shape[-2], shape[-1]])
        for i in range(P_index.shape[0]):
            P_index_matrix[P_index[i, 0], P_index[i, 1]] = 1

        P_index_matrix = np.stack([P_index_matrix, P_index_matrix], -1)
        P_index_matrix = np.expand_dims(P_index_matrix, 0)

        _theta = np.random.rand(grid.shape[0], grid.shape[1], grid.shape[2], grid.shape[3]) * (theta * 2) - theta
        _theta = _theta * P_index_matrix

        for i in range(_theta.shape[-1]):
            _theta[0, ..., i] = gaussian(_theta[0, ..., i], sigma=sigma)

        grid = grid + _theta

    return torch.from_numpy(grid)  # .to(torch.float32)


def generate_complex_data_with_motion(
    data: torch.Tensor,
    translation: Sequence[float] = (0, 0),
    rotation: int = 0,
    scale: float = 0,
    p: float = 0,
    theta: float = 0,
    sigma: float = 0,
    mode: str = "bilinear",
    align_corners: bool = True,
):
    """
    Generate complex data with motion.

    Parameters
    ----------
    data : torch.Tensor
        Data to be transformed.
    translation : Sequence[int, int]
        Translation of the grid.
    rotation : Sequence[int, int]
        Rotation of the grid.
    scale : float
        Scale of the grid.
    p : float
        Order of the polynomial.
    theta : float
        Angle of the polynomial.
    sigma : float
        Sigma of the polynomial.
    mode : str
        Interpolation mode.
    align_corners : bool
        Whether to align corners.
    """
    shape = data[:1, :, :, 0].shape

    affine_grid = generate_affine_grid(shape=shape, translation=translation, rotate=rotation, scale=scale).to(data)
    non_linear_grid = generate_nonlinear_grid(
        shape=shape,
        P=p,
        theta=theta,
        sigma=sigma,
    ).to(data)

    m_imspace_real = torch.nn.functional.grid_sample(
        data[..., 0].unsqueeze(0), affine_grid, mode=mode, align_corners=align_corners
    )
    m_imspace_real = torch.nn.functional.grid_sample(
        m_imspace_real, non_linear_grid, mode=mode, align_corners=align_corners
    )

    m_imspace_imag = torch.nn.functional.grid_sample(
        data[..., 1].unsqueeze(0), affine_grid, mode=mode, align_corners=align_corners
    )
    m_imspace_imag = torch.nn.functional.grid_sample(
        m_imspace_imag, non_linear_grid, mode=mode, align_corners=align_corners
    )

    motion_imspace = torch.cat([m_imspace_real, m_imspace_imag], 0).permute(1, 2, 3, 0)
    motion_imspace = motion_imspace[..., 0] + 1j * motion_imspace[..., 1]
    motion_imspace = torch.view_as_real(motion_imspace).squeeze(0)

    return motion_imspace
