# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Parts of the code have been taken from https://github.com/facebookresearch/fastMRI

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from mridc.collections.common.parts.fft import fft2c, ifft2c
from mridc.collections.common.parts.utils import complex_conj, complex_mul, to_tensor
from mridc.collections.reconstruction.data.subsample import MaskFunc
from mridc.collections.reconstruction.parts.utils import apply_mask, center_crop, complex_center_crop

__all__ = ["MRIDataTransforms"]


class MRIDataTransforms:
    """MRI preprocessing data transforms."""

    def __init__(
        self,
        mask_func: Optional[List[MaskFunc]] = None,
        shift_mask: bool = False,
        mask_center_scale: Optional[float] = 0.02,
        half_scan_percentage: float = 0.0,
        crop_size: Optional[Tuple[int, int]] = None,
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
            mask_center_scale: The scale of the center of the mask.
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
        self.mask_center_scale = mask_center_scale
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
        slice_idx: int,
    ) -> Tuple[
        Union[Union[List[Union[torch.Tensor, Any]], torch.Tensor], Any],
        Union[Optional[torch.Tensor], Any],
        Union[List, Any],
        Union[Optional[torch.Tensor], Any],
        Union[torch.Tensor, Any],
        str,
        int,
        Union[Union[List, torch.Tensor], Any],
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
            slice_idx: The slice number.

        Returns:
            The transformed data.
        """
        kspace = to_tensor(kspace)

        # This condition is necessary in case of auto estimation of sense maps.
        if sensitivity_map is not None and sensitivity_map.size != 0:
            sensitivity_map = to_tensor(sensitivity_map)

        # Apply zero-filling on kspace
        if self.kspace_zero_filling_size is not None and self.kspace_zero_filling_size not in ("", "None"):
            padding_top = np.floor_divide(abs(int(self.kspace_zero_filling_size[0]) - kspace.shape[1]), 2)
            padding_bottom = padding_top
            padding_left = np.floor_divide(abs(int(self.kspace_zero_filling_size[1]) - kspace.shape[2]), 2)
            padding_right = padding_left

            kspace = torch.view_as_complex(kspace)
            kspace = torch.nn.functional.pad(
                kspace, pad=(padding_left, padding_right, padding_top, padding_bottom), mode="constant", value=0
            )
            kspace = torch.view_as_real(kspace)

            sensitivity_map = fft2c(sensitivity_map, self.fft_type)
            sensitivity_map = torch.view_as_complex(sensitivity_map)
            sensitivity_map = torch.nn.functional.pad(
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
        if sensitivity_map is not None and sensitivity_map.size != 0:
            target = torch.sum(complex_mul(ifft2c(kspace, fft_type=self.fft_type), complex_conj(sensitivity_map)), 0)
            target = torch.view_as_complex(target)
        elif target is not None and target.size != 0:
            target = to_tensor(target)
        elif "target" in attrs or "target_rss" in attrs:
            target = torch.tensor(attrs["target"])
        else:
            raise ValueError("No target found")

        target = torch.abs(target / torch.max(torch.abs(target)))

        seed = None if not self.use_seed else tuple(map(ord, fname))
        acq_start = attrs["padding_left"] if "padding_left" in attrs else 0
        acq_end = attrs["padding_right"] if "padding_left" in attrs else 0

        # This should be outside of the condition because it needs to be returned in the end, even if cropping is off.
        # crop_size = torch.tensor([attrs["recon_size"][0], attrs["recon_size"][1]])
        crop_size = target.shape

        if self.crop_size is not None and self.crop_size not in ("", "None"):
            # Check for smallest size against the target shape.
            h = int(self.crop_size[0]) if int(self.crop_size[0]) <= target.shape[0] else target.shape[0]
            w = int(self.crop_size[1]) if int(self.crop_size[1]) <= target.shape[1] else target.shape[1]

            # Check for smallest size against the stored recon shape in metadata.
            if crop_size[0] != 0:
                h = h if h <= crop_size[0] else crop_size[0]
            if crop_size[1] != 0:
                w = w if w <= crop_size[1] else crop_size[1]

            self.crop_size = (int(h), int(w))

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
        if self.crop_size is not None and self.crop_size not in ("", "None") and self.crop_before_masking:
            kspace = (
                complex_center_crop(kspace, self.crop_size)
                if self.kspace_crop
                else fft2c(
                    complex_center_crop(ifft2c(kspace, fft_type=self.fft_type), self.crop_size), fft_type=self.fft_type
                )
            )

        if self.mask_func is not None:
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
                        center_scale=self.mask_center_scale,
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
                    self.mask_func[0],  # type: ignore
                    seed,
                    (acq_start, acq_end),
                    shift=self.shift_mask,
                    half_scan_percentage=self.half_scan_percentage,
                    center_scale=self.mask_center_scale,
                )
                mask = mask.byte()
        else:
            masked_kspace = kspace
            acc = torch.tensor([np.around(mask.size / mask.sum())]) if mask is not None else torch.tensor([1])

            if mask is not None:
                mask = torch.from_numpy(mask)

            if mask.shape[0] == masked_kspace.shape[-2]:
                mask = mask.permute(1, 0)
            elif (mask is None) or (mask.shape[0] != masked_kspace.shape[-3] or mask.shape[1] != masked_kspace.shape[-2]):
                mask = torch.ones([masked_kspace.shape[-3], masked_kspace.shape[-2]], dtype=torch.float32)

            if mask.ndim == 1:
                mask = np.expand_dims(mask, axis=0)

            if mask.shape[-2] == 1:  # 1D mask
                mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(-1)
            else:  # 2D mask
                # Crop loaded mask.
                if self.crop_size is not None and self.crop_size not in ("", "None"):
                    mask = center_crop(mask, self.crop_size)

                mask = mask.unsqueeze(0).unsqueeze(-1)

            if self.shift_mask:
                mask = torch.fft.fftshift(mask, dim=[-3, -2])

            masked_kspace = masked_kspace * mask
            mask = mask.byte()

        # Cropping after masking.
        if self.crop_size is not None and self.crop_size not in ("", "None") and not self.crop_before_masking:
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
                    elif self.fft_type == "fft_norm_only":
                        imspace = ifft2c(y, fft_type=self.fft_type)
                        masked_kspaces.append(fft2c(imspace, fft_type=self.fft_type))
                    elif self.fft_type == "backward_norm":
                        imspace = ifft2c(y, fft_type=self.fft_type, fft_normalization="backward")
                        masked_kspaces.append(fft2c(imspace, fft_type=self.fft_type, fft_normalization="backward"))
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
                elif self.fft_type == "fft_norm_only":
                    masked_kspace = fft2c(ifft2c(masked_kspace, fft_type=self.fft_type), fft_type=self.fft_type)
                elif self.fft_type == "backward_norm":
                    masked_kspace = fft2c(
                        ifft2c(masked_kspace, fft_type=self.fft_type, fft_normalization="backward"),
                        fft_type=self.fft_type,
                        fft_normalization="backward",
                    )
                else:
                    imspace = torch.fft.ifftn(torch.view_as_complex(masked_kspace), dim=[-2, -1], norm=None)
                    imspace = imspace / torch.max(torch.abs(imspace))
                    masked_kspace = torch.view_as_real(torch.fft.fftn(imspace, dim=[-2, -1], norm=None))

            if sensitivity_map.size != 0:
                sensitivity_map = sensitivity_map / torch.max(torch.abs(sensitivity_map))

            if eta.size != 0 and eta.ndim > 2:
                eta = eta / torch.max(torch.abs(eta))

            target = target / torch.max(torch.abs(target))

        return masked_kspace, sensitivity_map, mask, eta, target, fname, slice_idx, acc
