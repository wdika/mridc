# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Parts of the code have been taken from https://github.com/facebookresearch/fastMRI

from typing import Dict, Optional, Sequence, Tuple, Union, Any, List

import numpy as np
import torch
import torch.nn.functional as F
from numpy import ndarray
from torch import Tensor

from mridc import complex_mul, complex_conj, fft2c, ifft2c, rss, complex_abs
from .subsample import MaskFunc


def to_tensor(data: np.ndarray) -> torch.Tensor:
    """
    Converts a numpy array to a torch tensor.

    For complex arrays, the real and imaginary parts are stacked along the last
    dimension.

    Args:
        data: Input numpy array to be converted to torch.

    Returns:
        Torch tensor version of data.
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)

    return torch.from_numpy(data)


def tensor_to_complex_np(data: torch.Tensor) -> np.ndarray:
    """
    Converts a torch tensor to a numpy array.

    Args:
        data: Input torch tensor to be converted to numpy.

    Returns:
        Complex Numpy array version of data.
    """
    data = data.numpy()

    return data[..., 0] + 1j * data[..., 1]


def apply_mask(
    data: torch.Tensor,
    mask_func: MaskFunc,
    seed: Optional[Union[int, Tuple[int, ...]]] = None,
    padding: Optional[Sequence[int]] = None,
    shift: bool = False,
    half_scan_percentage: Optional[float] = 0.0,
) -> Tuple[Any, Any, Any]:
    """
    Subsample given k-space by multiplying with a mask.

    Args:
        data: The input k-space data. This should have at least 3 dimensions, where dimensions -3 and -2 are the
            spatial dimensions, and the final dimension has size 2 (for complex values).
        mask_func: A function that takes a shape (tuple of ints) and a random number seed and returns a mask.
        seed: Seed for the random number generator.
        padding: Padding value to apply for mask.
        shift: Toggle to shift mask when subsampling. Applicable on 2D data.
        half_scan_percentage: Percentage of kspace to be dropped.

    Returns:
        Tuple of subsampled k-space, mask, and mask indices.
    """
    shape = np.array(data.shape)
    shape[:-3] = 1
    mask, acc = mask_func(shape, seed, half_scan_percentage=half_scan_percentage)

    if padding is not None and padding[0] != 0:
        mask[:, :, : padding[0]] = 0
        mask[:, :, padding[1] :] = 0  # padding value inclusive on right of zeros

    if shift:
        mask = torch.fft.fftshift(mask, dim=(1, 2))

    masked_data = data * mask + 0.0  # the + 0.0 removes the sign of the zeros

    return masked_data, mask, acc


def mask_center(
    x: torch.Tensor, mask_from: Union[int, None], mask_to: Union[int, None], mask_type: str = "2D"
) -> torch.Tensor:
    """
    Apply a center crop to the input real image or batch of real images.

    Args:
        x: The input real image or batch of real images.
        mask_from: Part of center to start filling.
        mask_to: Part of center to end filling.
        mask_type: Type of mask to apply. Can be either "1D" or "2D".

    Returns:
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
    Initializes a mask with the center filled in.

    Can operate with different masks for each batch element.

    Args:
        x: The input real image or batch of real images.
        mask_from: Part of center to start filling.
        mask_to: Part of center to end filling.
        mask_type: Type of mask to apply. Can be either "1D" or "2D".

    Returns:
         A mask with the center filled.
    """
    if not mask_from.shape == mask_to.shape:
        raise ValueError("mask_from and mask_to must match shapes.")
    if not mask_from.ndim == 1:
        raise ValueError("mask_from and mask_to must have 1 dimension.")
    if not mask_from.shape[0] == 1 and (not x.shape[0] == mask_from.shape[0]) or (not x.shape[0] == mask_to.shape[0]):
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

    Args:
        data: The input tensor to be center cropped. It should have at least 2 dimensions and the cropping is applied
            along the last two dimensions.
        shape: The output shape. The shape should be smaller than the corresponding dimensions of data.

    Returns:
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

    Args:
        data: The complex input tensor to be center cropped. It should have at least 3 dimensions and the cropping is
            applied along dimensions -3 and -2 and the last dimensions should have a size of 2.
        shape: The output shape. The shape should be smaller than the corresponding dimensions of data.

    Returns:
        The center cropped image
    """
    if not (0 < shape[0] <= data.shape[-3] and 0 < shape[1] <= data.shape[-2]):
        raise ValueError("Invalid shapes.")

    w_from = torch.div((data.shape[-3] - shape[0]), 2, rounding_mode="trunc")
    h_from = torch.div((data.shape[-2] - shape[1]), 2, rounding_mode="trunc")
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to, :]  # type: ignore


def center_crop_to_smallest(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply a center crop on the larger image to the size of the smaller.

    The minimum is taken over dim=-1 and dim=-2. If x is smaller than y at dim=-1 and y is smaller than x at dim=-2,
        then the returned dimension will be a mixture of the two.

    Args:
        x: The first image.
        y: The second image.

    Returns:
        tuple of tensors x and y, each cropped to the minimum size.
    """
    smallest_width = min(x.shape[-1], y.shape[-1])
    smallest_height = min(x.shape[-2], y.shape[-2])
    x = center_crop(x, (smallest_height, smallest_width))
    y = center_crop(y, (smallest_height, smallest_width))

    return x, y


class UnetDataTransform:
    """A class that defines the data transform for the UNet."""

    def __init__(
        self,
        mask_func: Optional[MaskFunc] = None,
        shift_mask: bool = False,
        half_scan_percentage: float = 0.0,
        crop_size: Optional[Tuple] = None,
        kspace_crop: bool = False,
        crop_before_masking: bool = True,
        kspace_zero_filling_size: Optional[Tuple] = None,
        normalize_inputs: bool = False,
        fft_type: str = "orthogonal",
        output_type: str = "SENSE",
        use_seed: bool = True,
    ):
        """
        Args:
            mask_func: The function that will be used to mask the data.
            shift_mask: If True, the mask will be shifted to the center of the image.
            half_scan_percentage: The percentage of the image that will be masked.
            crop_size: The size of the crop.
            kspace_crop: If True, the data will be cropped to the center of the image.
            crop_before_masking: If True, the data will be cropped before masking.
            kspace_zero_filling_size: The size of padding in kspace -> zero filling.
            normalize_inputs: If True, the inputs will be normalized.
            fft_type: The type of FFT to use.
            output_type: The type of output to use.
            use_seed: If True, the seed will be set.
        """
        self.mask_func = mask_func
        self.shift_mask = shift_mask
        self.half_scan_percentage = half_scan_percentage

        self.crop_size = crop_size
        self.crop_before_masking = crop_before_masking
        self.kspace_zero_filling_size = kspace_zero_filling_size
        self.kspace_crop = kspace_crop

        self.normalize_inputs = normalize_inputs

        self.fft_type = fft_type

        self.output_type = output_type

        self.use_seed = use_seed

    def __call__(
        self,
        kspace: np.ndarray,
        sensitivity_map: np.ndarray,
        mask: np.ndarray,
        eta: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> Tuple[Union[Tensor, Any], Union[Tensor, Any], str, int, Union[Tensor, Any], ndarray, Tensor]:
        """
        Args:
            kspace: The kspace data.
            sensitivity_map: The sensitivity map.
            mask: The mask.
            eta: The initial estimation.
            target: The target.
            attrs: The attributes.
            fname: The filename.
            slice_num: The slice number.

        Returns:
            The transformed data.
        """
        kspace = to_tensor(kspace)

        # This condition is necessary in case of auto estimation of sense maps.
        if sensitivity_map is not None and sensitivity_map.size != 0:
            sensitivity_map = to_tensor(sensitivity_map)

        # Apply zero-filling on kspace
        if self.kspace_zero_filling_size is not None and self.kspace_zero_filling_size != "":
            # (padding_left,padding_right, padding_top,padding_bottom)
            padding_top = abs(int(self.kspace_zero_filling_size[0]) - kspace.shape[1]) // 2
            padding_bottom = padding_top
            padding_left = abs(int(self.kspace_zero_filling_size[1]) - kspace.shape[2]) // 2
            padding_right = padding_left

            kspace = torch.view_as_complex(kspace)
            kspace = F.pad(
                kspace, pad=(padding_left, padding_right, padding_top, padding_bottom), mode="constant", value=0
            )
            kspace = torch.view_as_real(kspace)

            sensitivity_map = fft2c(sensitivity_map, self.fft_type)
            sensitivity_map = torch.view_as_complex(sensitivity_map)
            sensitivity_map = F.pad(
                sensitivity_map,
                pad=(padding_left, padding_right, padding_top, padding_bottom),
                mode="constant",
                value=0,
            )
            sensitivity_map = torch.view_as_real(sensitivity_map)
            sensitivity_map = ifft2c(sensitivity_map, self.fft_type)

        # TODO: add RSS target option
        target = torch.sum(complex_mul(ifft2c(kspace, fft_type=self.fft_type), complex_conj(sensitivity_map)), 0)
        target = torch.abs(target[..., 0] + 1j * target[..., 1])
        target = target / torch.max(target)

        seed = None if not self.use_seed else tuple(map(ord, fname))
        acq_start = attrs["padding_left"] if "padding_left" in attrs else 0
        acq_end = attrs["padding_right"] if "padding_left" in attrs else 0

        # This should be outside of the condition because it needs to be returned in the end, even if cropping is off.
        crop_size = torch.tensor([attrs["recon_size"][0], attrs["recon_size"][1]])

        if self.crop_size is not None:
            # Check for smallest size against the target shape.
            h = int(self.crop_size[0]) if int(self.crop_size[0]) <= target.shape[0] else target.shape[0]
            w = int(self.crop_size[1]) if int(self.crop_size[1]) <= target.shape[1] else target.shape[1]

            # Check for smallest size against the stored recon shape in metadata.
            if crop_size[0] != 0:
                h = h if h <= crop_size[0] else crop_size[0]
            if crop_size[1] != 0:
                w = w if w <= crop_size[1] else crop_size[1]

            self.crop_size = (h, w)
            crop_size = torch.tensor([self.crop_size[0], self.crop_size[1]])

            target = center_crop(target, self.crop_size)

            if sensitivity_map is not None and sensitivity_map.size != 0:
                sensitivity_map = (
                    ifft2c(
                        complex_center_crop(fft2c(sensitivity_map, fft_type=self.fft_type), self.crop_size),
                        fft_type=self.fft_type,
                    )
                    if self.kspace_crop
                    else complex_center_crop(sensitivity_map, self.crop_size)
                )

        # Cropping before masking will maintain the shape of original kspace intact for masking.
        if self.crop_size is not None and self.crop_before_masking:
            kspace = (
                complex_center_crop(kspace, self.crop_size)
                if self.kspace_crop
                else fft2c(
                    complex_center_crop(ifft2c(kspace, fft_type=self.fft_type), self.crop_size), fft_type=self.fft_type
                )
            )

        if self.mask_func:
            # Check for multiple masks/accelerations.
            if isinstance(self.mask_func, list):
                masked_kspaces = []
                accs = []
                for m in self.mask_func:
                    _masked_kspace, _, _acc = apply_mask(
                        kspace,
                        m,
                        seed,
                        (acq_start, acq_end),
                        shift=self.shift_mask,
                        half_scan_percentage=self.half_scan_percentage,
                    )
                    masked_kspaces.append(_masked_kspace)
                    accs.append(_acc)
                masked_kspace = masked_kspaces
                acc = accs
            else:
                masked_kspace, mask, acc = apply_mask(
                    kspace,
                    self.mask_func,
                    seed,
                    (acq_start, acq_end),
                    shift=self.shift_mask,
                    half_scan_percentage=self.half_scan_percentage,
                )
        else:
            masked_kspace = kspace
            acc = (
                torch.tensor([np.around(mask.size / mask.sum())]) if mask is not None else torch.tensor([1])
            )  # type: ignore

        # Cropping after masking.
        if self.crop_size is not None and not self.crop_before_masking:
            masked_kspace = (
                complex_center_crop(masked_kspace, self.crop_size)  # type: ignore
                if self.kspace_crop
                else fft2c(
                    complex_center_crop(ifft2c(masked_kspace, fft_type=self.fft_type), self.crop_size),
                    fft_type=self.fft_type,
                )
            )

        # Normalize by the max value.
        if self.normalize_inputs:
            if sensitivity_map.size != 0:
                sensitivity_map = sensitivity_map / torch.max(torch.abs(sensitivity_map))

            target = target / torch.max(torch.abs(target))

        if isinstance(self.mask_func, list):
            images = []
            for y in masked_kspace:
                imspace = (
                    ifft2c(y, fft_type=self.fft_type)
                    if self.fft_type in ("orthogonal", "orthogonal_norm_only")
                    else torch.view_as_real(torch.fft.ifftn(torch.view_as_complex(y), dim=[-2, -1], norm=None))
                )

                if self.normalize_inputs:
                    imspace = imspace / torch.max(torch.abs(imspace))

                if self.output_type == "SENSE":
                    image = complex_mul(imspace, complex_conj(sensitivity_map)).sum(dim=0)
                elif self.output_type == "RSS":
                    image = complex_abs(rss(imspace))
                else:
                    raise NotImplementedError("Output type can be either SENSE or RSS")

                images.append(image)

            image = images
        else:
            imspace = (
                ifft2c(masked_kspace, fft_type=self.fft_type)
                if self.fft_type in ("orthogonal", "orthogonal_norm_only")
                else torch.view_as_real(torch.fft.ifftn(torch.view_as_complex(masked_kspace), dim=[-2, -1], norm=None))
            )

            if self.normalize_inputs:
                imspace = imspace / torch.max(torch.abs(imspace))

            if self.output_type == "SENSE":
                image = complex_mul(imspace, complex_conj(sensitivity_map)).sum(dim=0)
            elif self.output_type == "RSS":
                image = complex_abs(rss(imspace))
            else:
                raise NotImplementedError("Output type can be either SENSE or RSS")

        # This is needed when using the ssim as loss function.
        max_value = np.array(torch.max(torch.abs(target)).item()).astype(np.float32)

        return image, target, fname, slice_num, acc, max_value, crop_size


class PhysicsInformedDataTransform:
    """A class that defines the data transform for the physics informed methods."""

    def __init__(
        self,
        mask_func: Optional[MaskFunc] = None,
        shift_mask: bool = False,
        half_scan_percentage: float = 0.0,
        crop_size: Optional[Tuple] = None,
        kspace_crop: bool = False,
        crop_before_masking: bool = True,
        kspace_zero_filling_size: Optional[Tuple] = None,
        normalize_inputs: bool = False,
        fft_type: str = "orthogonal",
        use_seed: bool = True,
    ):
        """
        Initialize the data transform.

        Args:
            mask_func: The function that masks the kspace.
            shift_mask: Whether to shift the mask.
            half_scan_percentage: The percentage of the scan to be used.
            crop_size: The size of the crop.
            kspace_crop: Whether to crop the kspace.
            crop_before_masking: Whether to crop before masking.
            kspace_zero_filling_size: The size of padding in kspace -> zero filling.
            normalize_inputs: Whether to normalize the inputs.
            fft_type: The type of the FFT.
            use_seed: Whether to use the seed.
        """
        self.mask_func = mask_func
        self.shift_mask = shift_mask
        self.half_scan_percentage = half_scan_percentage
        self.crop_size = crop_size
        self.kspace_crop = kspace_crop
        self.crop_before_masking = crop_before_masking
        self.kspace_zero_filling_size = kspace_zero_filling_size
        self.normalize_inputs = normalize_inputs
        self.fft_type = fft_type
        self.use_seed = use_seed

    def __call__(
        self,
        kspace: np.ndarray,
        sensitivity_map: np.ndarray,
        mask: np.ndarray,
        eta: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> Tuple[
        Union[Union[List[Union[Union[float, Tensor], Any]], Tensor, float], Any],
        Union[Optional[Tensor], Any],
        Union[list, Any],
        Union[Optional[Tensor], Any],
        Union[Tensor, Any],
        str,
        int,
        Union[Union[list, Tensor], Any],
        Any,
        Tensor,
    ]:
        """
        Apply the data transform.

        Args:
            kspace: The kspace.
            sensitivity_map: The sensitivity map.
            mask: The mask.
            eta: The initial estimation.
            target: The target.
            attrs: The attributes.
            fname: The file name.
            slice_num: The slice number.

        Returns:
            The transformed data.
        """
        kspace = to_tensor(kspace)

        # This condition is necessary in case of auto estimation of sense maps.
        if sensitivity_map is not None and sensitivity_map.size != 0:
            sensitivity_map = to_tensor(sensitivity_map)

        # Apply zero-filling on kspace
        if self.kspace_zero_filling_size is not None and self.kspace_zero_filling_size != "":
            # (padding_left,padding_right, padding_top,padding_bottom)
            padding_top = abs(int(self.kspace_zero_filling_size[0]) - kspace.shape[1]) // 2
            padding_bottom = padding_top
            padding_left = abs(int(self.kspace_zero_filling_size[1]) - kspace.shape[2]) // 2
            padding_right = padding_left

            kspace = torch.view_as_complex(kspace)
            kspace = F.pad(
                kspace, pad=(padding_left, padding_right, padding_top, padding_bottom), mode="constant", value=0
            )
            kspace = torch.view_as_real(kspace)

            sensitivity_map = fft2c(sensitivity_map, self.fft_type)
            sensitivity_map = torch.view_as_complex(sensitivity_map)
            sensitivity_map = F.pad(
                sensitivity_map,
                pad=(padding_left, padding_right, padding_top, padding_bottom),
                mode="constant",
                value=0,
            )
            sensitivity_map = torch.view_as_real(sensitivity_map)
            sensitivity_map = ifft2c(sensitivity_map, self.fft_type)

        if eta is not None and eta.size != 0:
            eta = to_tensor(eta)
        else:
            eta = torch.tensor([])

        # TODO: add RSS target option
        target = torch.sum(complex_mul(ifft2c(kspace, fft_type=self.fft_type), complex_conj(sensitivity_map)), 0)
        target = torch.view_as_complex(target)
        target = torch.abs(target / torch.max(torch.abs(target)))

        seed = None if not self.use_seed else tuple(map(ord, fname))
        acq_start = attrs["padding_left"] if "padding_left" in attrs else 0
        acq_end = attrs["padding_right"] if "padding_left" in attrs else 0

        # This should be outside of the condition because it needs to be returned in the end, even if cropping is off.
        # crop_size = torch.tensor([attrs["recon_size"][0], attrs["recon_size"][1]])
        crop_size = target.shape

        if self.crop_size is not None:
            # Check for smallest size against the target shape.
            h = int(self.crop_size[0]) if int(self.crop_size[0]) <= target.shape[0] else target.shape[0]
            w = int(self.crop_size[1]) if int(self.crop_size[1]) <= target.shape[1] else target.shape[1]

            # Check for smallest size against the stored recon shape in metadata.
            if crop_size[0] != 0:
                h = h if h <= crop_size[0] else crop_size[0]
            if crop_size[1] != 0:
                w = w if w <= crop_size[1] else crop_size[1]

            self.crop_size = (h, w)
            crop_size = torch.tensor([self.crop_size[0], self.crop_size[1]])

            target = center_crop(target, self.crop_size)
            if sensitivity_map is not None and sensitivity_map.size != 0:
                sensitivity_map = (
                    ifft2c(
                        complex_center_crop(fft2c(sensitivity_map, fft_type=self.fft_type), self.crop_size),
                        fft_type=self.fft_type,
                    )
                    if self.kspace_crop
                    else complex_center_crop(sensitivity_map, self.crop_size)
                )

            if eta is not None and eta.ndim > 2:
                eta = (
                    ifft2c(
                        complex_center_crop(fft2c(eta, fft_type=self.fft_type), self.crop_size), fft_type=self.fft_type
                    )
                    if self.kspace_crop
                    else complex_center_crop(eta, self.crop_size)
                )

        # Cropping before masking will maintain the shape of original kspace intact for masking.
        if self.crop_size is not None and self.crop_before_masking:
            kspace = (
                complex_center_crop(kspace, self.crop_size)
                if self.kspace_crop
                else fft2c(
                    complex_center_crop(ifft2c(kspace, fft_type=self.fft_type), self.crop_size), fft_type=self.fft_type
                )
            )

        if self.mask_func:
            # Check for multiple masks/accelerations.
            if isinstance(self.mask_func, list):
                masked_kspaces = []
                masks = []
                accs = []
                for m in self.mask_func:
                    _masked_kspace, _mask, _acc = apply_mask(
                        kspace,
                        m,
                        seed,
                        (acq_start, acq_end),
                        shift=self.shift_mask,
                        half_scan_percentage=self.half_scan_percentage,
                    )
                    masked_kspaces.append(_masked_kspace)
                    masks.append(_mask.byte())
                    accs.append(_acc)
                masked_kspace = masked_kspaces
                mask = masks
                acc = accs
            else:
                masked_kspace, mask, acc = apply_mask(
                    kspace,
                    self.mask_func,
                    seed,
                    (acq_start, acq_end),
                    shift=self.shift_mask,
                    half_scan_percentage=self.half_scan_percentage,
                )
                mask = mask.byte()
        else:
            masked_kspace = kspace
            acc = torch.tensor([np.around(mask.size / mask.sum())]) if mask is not None else torch.tensor([1])

            if mask.shape[-2] == 1:  # 1D mask
                shape = np.array(kspace.shape)
                num_cols = shape[-2]
                shape[:-3] = 1
                mask_shape = [1] * len(shape)
                mask_shape[-2] = num_cols
                mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))
                mask = mask.reshape(*mask_shape)
                mask[:, :, :acq_start] = 0
                mask[:, :, acq_end:] = 0
            else:  # 2D mask
                mask = torch.from_numpy(mask.astype(np.float32))

                # Crop loaded mask.
                if self.crop_size is not None:
                    mask = center_crop(mask, self.crop_size)

                mask = mask.unsqueeze(0).unsqueeze(-1)

            if self.shift_mask:
                mask = torch.fft.fftshift(mask, dim=[-3, -2])

            masked_kspace = masked_kspace * mask

            mask = mask.byte()

        # Cropping after masking.
        if self.crop_size is not None and not self.crop_before_masking:
            masked_kspace = (
                complex_center_crop(masked_kspace, self.crop_size)
                if self.kspace_crop
                else fft2c(
                    complex_center_crop(ifft2c(masked_kspace, fft_type=self.fft_type), self.crop_size),
                    fft_type=self.fft_type,
                )
            )

            mask = center_crop(mask.squeeze(-1), self.crop_size).unsqueeze(-1)

        # Normalize by the max value.
        if self.normalize_inputs:
            if isinstance(self.mask_func, list):
                masked_kspaces = []
                for y in masked_kspace:
                    if self.fft_type in ("orthogonal", "orthogonal_norm_only"):
                        imspace = ifft2c(y, fft_type=self.fft_type)
                        imspace = imspace / torch.max(torch.abs(imspace))
                        masked_kspaces.append(fft2c(imspace, fft_type=self.fft_type))
                    else:
                        imspace = torch.fft.ifftn(torch.view_as_complex(y), dim=[-2, -1], norm=None)
                        imspace = imspace / torch.max(torch.abs(imspace))
                        masked_kspaces.append(torch.view_as_real(torch.fft.fftn(imspace, dim=[-2, -1], norm=None)))
                masked_kspace = masked_kspaces
            else:
                if self.fft_type in ("orthogonal", "orthogonal_norm_only"):
                    imspace = ifft2c(masked_kspace, fft_type=self.fft_type)
                    imspace = imspace / torch.max(torch.abs(imspace))
                    masked_kspace = fft2c(imspace, fft_type=self.fft_type)
                else:
                    imspace = torch.fft.ifftn(torch.view_as_complex(masked_kspace), dim=[-2, -1], norm=None)
                    imspace = imspace / torch.max(torch.abs(imspace))
                    masked_kspace = torch.view_as_real(torch.fft.fftn(imspace, dim=[-2, -1], norm=None))

            if sensitivity_map.size != 0:
                sensitivity_map = sensitivity_map / torch.max(torch.abs(sensitivity_map))

            if eta.size != 0 and eta.ndim > 2:
                eta = eta / torch.max(torch.abs(eta))

            target = target / torch.max(torch.abs(target))

        # This is needed when using the ssim as loss function.
        max_value = np.array(torch.max(torch.abs(target)).item()).astype(np.float32)

        return (masked_kspace, sensitivity_map, mask, eta, target, fname, slice_num, acc, max_value, crop_size)
