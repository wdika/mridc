# coding=utf-8
__author__ = "Dimitrios Karkalousos"

# Parts of the code have been taken from https://github.com/facebookresearch/fastMRI

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

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
    "apply_mask",
    "mask_center",
    "batched_mask_center",
    "center_crop",
    "complex_center_crop",
    "center_crop_to_smallest",
]

from mridc.collections.reconstruction.data.subsample import MaskFunc


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


def apply_mask(
    data: torch.Tensor,
    mask_func: MaskFunc,
    seed: Optional[Union[int, Tuple[int, ...]]] = None,
    padding: Optional[Sequence[int]] = None,
    shift: bool = False,
    half_scan_percentage: Optional[float] = 0.0,
    center_scale: Optional[float] = 0.02,
    existing_mask: Optional[torch.Tensor] = None,
) -> Tuple[Any, Any, int]:
    """
    Subsample given k-space by multiplying with a mask.

    Parameters
    ----------
    data: The input k-space data. This should have at least 3 dimensions, where dimensions -3 and -2 are the
        spatial dimensions, and the final dimension has size 2 (for complex values).
    mask_func: A function that takes a shape (tuple of ints) and a random number seed and returns a mask.
    seed: Seed for the random number generator.
    padding: Padding value to apply for mask.
    shift: Toggle to shift mask when subsampling. Applicable on 2D data.
    half_scan_percentage: Percentage of kspace to be dropped.
    center_scale: Scale of the center of the mask. Applicable on Gaussian masks.
    existing_mask: When given, use this mask instead of generating a new one.

    Returns
    -------
    Tuple of subsampled k-space, mask, and mask indices.
    """
    shape = np.array(data.shape)
    shape[:-3] = 1

    if existing_mask is None:
        mask, acc = mask_func(shape, seed, half_scan_percentage=half_scan_percentage, scale=center_scale)
    else:
        mask = existing_mask
        acc = mask.size / mask.sum()

    mask = mask.to(data.device)

    if padding is not None and padding[0] != 0:
        mask[:, :, : padding[0]] = 0
        # padding value inclusive on right of zeros
        mask[:, :, padding[1] :] = 0

    if shift:
        mask = torch.fft.fftshift(mask, dim=(1, 2))

    masked_data = data * mask + 0.0  # the + 0.0 removes the sign of the zeros

    return masked_data, mask, acc


def mask_center(
    x: torch.Tensor, mask_from: Optional[int], mask_to: Optional[int], mask_type: str = "2D"
) -> torch.Tensor:
    """
    Apply a center crop to the input real image or batch of real images.

    Parameters
    ----------
    x: The input real image or batch of real images.
    mask_from: Part of center to start filling.
    mask_to: Part of center to end filling.
    mask_type: Type of mask to apply. Can be either "1D" or "2D".

    Returns
    -------
     A mask with the center filled.
    """
    mask = torch.zeros_like(x)

    if isinstance(mask_from, list):
        mask_from = mask_from[0]

    if isinstance(mask_to, list):
        mask_to = mask_to[0]

    if mask_type == "1D":
        mask[:, :, :, mask_from:mask_to] = x[:, :, :, mask_from:mask_to]
    elif mask_type == "2D":
        mask[:, :, mask_from:mask_to] = x[:, :, mask_from:mask_to]

    return mask


def batched_mask_center(
    x: torch.Tensor, mask_from: torch.Tensor, mask_to: torch.Tensor, mask_type: str = "2D"
) -> torch.Tensor:
    """
    Initializes a mask with the center filled in. Can operate with different masks for each batch element.

    Parameters
    ----------
        x: The input real image or batch of real images.
        mask_from: Part of center to start filling.
        mask_to: Part of center to end filling.
        mask_type: Type of mask to apply. Can be either "1D" or "2D".

    Returns
    -------
     A mask with the center filled.
    """
    if mask_from.shape != mask_to.shape:
        raise ValueError("mask_from and mask_to must match shapes.")
    if mask_from.ndim != 1:
        raise ValueError("mask_from and mask_to must have 1 dimension.")
    if mask_from.shape[0] not in (1, x.shape[0]) or x.shape[0] != mask_to.shape[0]:
        raise ValueError("mask_from and mask_to must have batch_size length.")

    if mask_from.shape[0] == 1:
        mask = mask_center(x, int(mask_from), int(mask_to), mask_type=mask_type)
    else:
        mask = torch.zeros_like(x)
        for i, (start, end) in enumerate(zip(mask_from, mask_to)):
            mask[i, :, :, start:end] = x[i, :, :, start:end]

    return mask


def center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Apply a center crop to the input real image or batch of real images.

    Parameters
    ----------
    data: The input tensor to be center cropped. It should have at least 2 dimensions and the cropping is applied
        along the last two dimensions.
    shape: The output shape. The shape should be smaller than the corresponding dimensions of data.

    Returns
    -------
    The center cropped image.
    """
    if not (0 < shape[0] <= data.shape[-2] and 0 < shape[1] <= data.shape[-1]):
        raise ValueError("Invalid shapes.")

    w_from = torch.div((data.shape[-2] - shape[0]), 2, rounding_mode="trunc")
    h_from = torch.div((data.shape[-1] - shape[1]), 2, rounding_mode="trunc")
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to]  # type: ignore


def complex_center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Apply a center crop to the input image or batch of complex images.

    Parameters
    ----------
    data: The complex input tensor to be center cropped. It should have at least 3 dimensions and the cropping is
        applied along dimensions -3 and -2 and the last dimensions should have a size of 2.
    shape: The output shape. The shape should be smaller than the corresponding dimensions of data.

    Returns
    -------
    The center cropped image.
    """
    if not (0 < shape[0] <= data.shape[-3] and 0 < shape[1] <= data.shape[-2]):
        raise ValueError("Invalid shapes.")

    w_from = torch.div((data.shape[-3] - shape[0]), 2, rounding_mode="trunc")
    h_from = torch.div((data.shape[-2] - shape[1]), 2, rounding_mode="trunc")
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to, :]  # type: ignore


def center_crop_to_smallest(
    x: Union[torch.Tensor, np.ndarray], y: Union[torch.Tensor, np.ndarray]
) -> Tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]]:
    """
    Apply a center crop on the larger image to the size of the smaller.

    The minimum is taken over dim=-1 and dim=-2. If x is smaller than y at dim=-1 and y is smaller than x at dim=-2,
        then the returned dimension will be a mixture of the two.

    Parameters
    ----------
    x: The first image.
    y: The second image.

    Returns
    -------
    Tuple of tensors x and y, each cropped to the minimum size.
    """
    smallest_width = min(x.shape[-1], y.shape[-1])
    smallest_height = min(x.shape[-2], y.shape[-2])
    x = center_crop(x, (smallest_height, smallest_width))
    y = center_crop(y, (smallest_height, smallest_width))

    return x, y
