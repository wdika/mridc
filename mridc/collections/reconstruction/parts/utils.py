# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Parts of the code have been taken from https://github.com/facebookresearch/fastMRI

from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import torch

__all__ = [
    "apply_mask",
    "mask_center",
    "batched_mask_center",
    "center_crop",
    "complex_center_crop",
    "center_crop_to_smallest",
]

from mridc.collections.reconstruction.data.subsample import MaskFunc


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
        mask[:, :, padding[1] :] = 0  # padding value inclusive on right of zeros

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
