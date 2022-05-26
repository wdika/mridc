# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Parts of the code have been taken from https://github.com/facebookresearch/fastMRI

from pathlib import Path
from typing import Any, Dict, Union

import h5py
import numpy as np
import torch

__all__ = [
    "is_none",
    "to_tensor",
    "tensor_to_complex_np",
    "complex_mul",
    "complex_conj",
    "complex_abs",
    "complex_abs_sq",
    "rss",
    "rss_complex",
    "sense",
    "coil_combination",
    "save_reconstructions",
    "check_stacked_complex",
]


def is_none(x: Union[Any, None]) -> bool:
    """
    Check if a string is None.

    Parameters
    ----------
    x: The string to check.

    Returns
    -------
    True if x is None, False otherwise.
    """
    return x is None or str(x).lower() == "none"


def to_tensor(data: np.ndarray) -> torch.Tensor:
    """
    Converts a numpy array to a torch tensor.

    For complex arrays, the real and imaginary parts are stacked along the last
    dimension.

    Parameters
    ----------
    data: Input numpy array to be converted to torch.

    Returns
    -------
    Torch tensor version of data.
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)

    return torch.from_numpy(data)


def tensor_to_complex_np(data: torch.Tensor) -> np.ndarray:
    """
    Converts a torch tensor to a numpy array.

    Parameters
    ----------
    data: Input torch tensor to be converted to numpy.

    Returns
    -------
    Complex Numpy array version of data.
    """
    data = data.numpy()

    return data[..., 0] + 1j * data[..., 1]


def reshape_fortran(x, shape) -> torch.Tensor:
    """Reshapes a tensor in Fortran order. Taken from https://stackoverflow.com/a/63964246"""
    return x.permute(*reversed(range(len(x.shape)))).reshape(*reversed(shape)).permute(*reversed(range(len(shape))))


def complex_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Complex multiplication.

    This multiplies two complex tensors assuming that they are both stored as
    real arrays with the last dimension being the complex dimension.

    Parameters
    ----------
    x: A PyTorch tensor with the last dimension of size 2.
    y: A PyTorch tensor with the last dimension of size 2.

    Returns
    -------
    A PyTorch tensor with the last dimension of size 2.
    """
    if not x.shape[-1] == y.shape[-1] == 2:
        raise ValueError("Tensors do not have separate complex dim.")

    re = x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1]
    im = x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0]

    return torch.stack((re, im), dim=-1)


def complex_conj(x: torch.Tensor) -> torch.Tensor:
    """
    Complex conjugate.

    This applies the complex conjugate assuming that the input array has the
    last dimension as the complex dimension.

    Parameters
    ----------
    x: A PyTorch tensor with the last dimension of size 2.

    Returns
    -------
    A PyTorch tensor with the last dimension of size 2.
    """
    if x.shape[-1] != 2:
        raise ValueError("Tensor does not have separate complex dim.")

    return torch.stack((x[..., 0], -x[..., 1]), dim=-1)


def complex_abs(data: torch.Tensor) -> torch.Tensor:
    """
    Compute the absolute value of a complex valued input tensor.

    Parameters
    ----------
    data: A complex valued tensor, where the size of the final dimension should be 2.

    Returns
    -------
    Absolute value of data.
    """
    if data.shape[-1] != 2:
        raise ValueError("Tensor does not have separate complex dim.")

    return (data**2).sum(dim=-1).sqrt()


def complex_abs_sq(data: torch.Tensor) -> torch.Tensor:
    """
    Compute the squared absolute value of a complex tensor.

    Parameters
    ----------
    data: A complex valued tensor, where the size of the final dimension should be 2.

    Returns
    -------
    Squared absolute value of data.
    """
    if data.shape[-1] != 2:
        raise ValueError("Tensor does not have separate complex dim.")

    return (data**2).sum(dim=-1)


def check_stacked_complex(data: torch.Tensor) -> torch.Tensor:
    """
    Check if tensor is stacked complex (real & imag parts stacked along last dim) and convert it to a combined complex
    tensor.

    Parameters
    ----------
    data: A complex valued tensor, where the size of the final dimension might be 2.

    Returns
    -------
    A complex valued tensor.
    """
    return torch.view_as_complex(data) if data.shape[-1] == 2 else data


def rss(data: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Compute the Root Sum of Squares (RSS).

    RSS is computed assuming that dim is the coil dimension.

    Parameters
    ----------
    data: The input tensor
    dim: The dimensions along which to apply the RSS transform

    Returns
    -------
    The RSS value.
    """
    return torch.sqrt((data**2).sum(dim))


def rss_complex(data: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Compute the Root Sum of Squares (RSS) for complex inputs.

    RSS is computed assuming that dim is the coil dimension.

    Parameters
    ----------
    data: The input tensor
    dim: The dimensions along which to apply the RSS transform

    Returns
    -------
    The RSS value.
    """
    return torch.sqrt(complex_abs_sq(data).sum(dim))


def sense(data: torch.Tensor, sensitivity_maps: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    The SENSitivity Encoding (SENSE) transform [1]_.

    References
    ----------
    .. [1] Pruessmann KP, Weiger M, Scheidegger MB, Boesiger P. SENSE: Sensitivity encoding for fast MRI. Magn Reson Med 1999; 42:952-962.

    Parameters
    ----------
    data: The input tensor
    sensitivity_maps: The sensitivity maps
    dim: The coil dimension

    Returns
    -------
    A coil-combined image.
    """
    return complex_mul(data, complex_conj(sensitivity_maps)).sum(dim)


def coil_combination(
    data: torch.Tensor, sensitivity_maps: torch.Tensor, method: str = "SENSE", dim: int = 0
) -> torch.Tensor:
    """
    Coil combination.

    Parameters
    ----------
    data: The input tensor.
    sensitivity_maps: The sensitivity maps.
    method: The coil combination method.
    dim: The dimensions along which to apply the coil combination transform.

    Returns
    -------
    Coil combined data.
    """
    if method == "SENSE":
        return sense(data, sensitivity_maps, dim)
    if method == "RSS":
        return rss(data, dim)
    raise ValueError("Output type not supported.")


def save_reconstructions(reconstructions: Dict[str, np.ndarray], out_dir: Path):
    """
    Save reconstruction images.

    This function writes to h5 files that are appropriate for submission to the
    leaderboard.

    Parameters
    ----------
    reconstructions: A dictionary mapping input filenames to corresponding reconstructions.
    out_dir: Path to the output directory where the reconstructions should be saved.
    """
    out_dir.mkdir(exist_ok=True, parents=True)
    for fname, recons in reconstructions.items():
        with h5py.File(out_dir / fname, "w") as hf:
            hf.create_dataset("reconstruction", data=recons)
