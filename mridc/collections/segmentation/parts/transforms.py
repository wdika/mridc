# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from mridc.collections.common.parts.fft import fft2, ifft2
from mridc.collections.common.parts.utils import (
    is_none,
    rss,
    sense,
    to_tensor,
)
from mridc.collections.reconstruction.data.subsample import MaskFunc
from mridc.collections.reconstruction.parts.transforms import (
    GeometricDecompositionCoilCompression,
    NoisePreWhitening,
)
from mridc.collections.reconstruction.parts.utils import apply_mask, center_crop, complex_center_crop

__all__ = ["JRSMRIDataTransforms"]


class JRSMRIDataTransforms:
    """Joint Reconstruction & Segmentation preprocessing data transforms."""

    def __init__(
        self,
        complex_data: bool = True,
        apply_prewhitening: bool = False,
        prewhitening_scale_factor: float = 1.0,
        prewhitening_patch_start: int = 10,
        prewhitening_patch_length: int = 30,
        apply_gcc: bool = False,
        gcc_virtual_coils: int = 10,
        gcc_calib_lines: int = 24,
        gcc_align_data: bool = True,
        coil_combination_method: str = "SENSE",
        dimensionality: int = 2,
        mask_func: Optional[List[MaskFunc]] = None,
        shift_mask: bool = False,
        mask_center_scale: Optional[float] = 0.02,
        half_scan_percentage: float = 0.0,
        remask: bool = False,
        crop_size: Optional[Tuple[int, int]] = None,
        kspace_crop: bool = False,
        crop_before_masking: bool = True,
        kspace_zero_filling_size: Optional[Tuple] = None,
        normalize_inputs: bool = False,
        max_norm: bool = True,
        fft_centered: bool = True,
        fft_normalization: str = "ortho",
        spatial_dims: Sequence[int] = None,
        coil_dim: int = 0,
        consecutive_slices: int = 1,
        use_seed: bool = True,
    ):
        """
        Initialize the data transform.

        Parameters
        ----------
        complex_data: bool
            Whether the input data is complex or not.
        apply_prewhitening: Apply prewhitening.
            bool
        prewhitening_scale_factor: Prewhitening scale factor.
            float
        prewhitening_patch_start: Prewhitening patch start.
            int
        prewhitening_patch_length: Prewhitening patch length.
            int
        apply_gcc: Apply Geometric Decomposition Coil Compression.
            bool
        gcc_virtual_coils: GCC virtual coils.
            int
        gcc_calib_lines: GCC calibration lines.
            int
        gcc_align_data: GCC align data.
            bool
        coil_combination_method: Coil combination method. Default: SENSE.
            str
        dimensionality: Dimensionality.
            int
        mask_func: Mask function.
            List[MaskFunc]
        shift_mask: Shift mask.
            bool
        mask_center_scale: Mask center scale.
            float
        half_scan_percentage: Half scan percentage.
            float
        remask: Use the same mask. Default: False.
            bool
        crop_size: Crop size.
            Tuple[int, int]
        kspace_crop: K-space crop.
            bool
        crop_before_masking: Crop before masking.
            bool
        kspace_zero_filling_size: K-space zero filling size.
            Tuple
        normalize_inputs: Normalize inputs.
            bool
        max_norm: Normalization by the maximum value.
            bool
        fft_centered: FFT centered.
            bool
        fft_normalization: FFT normalization.
            str
        spatial_dims: Spatial dimensions.
            Sequence[int]
        coil_dim: Coil dimension.
            int
        consecutive_slices: Number of consecutive slices.
            int
        use_seed: Use seed.
            bool
        """
        self.complex_data = complex_data
        if self.complex_data:
            self.coil_combination_method = coil_combination_method
        elif not is_none(coil_combination_method):
            raise ValueError("Coil combination method for non-complex data should be None.")
        self.dimensionality = dimensionality
        if self.complex_data:
            self.mask_func = mask_func
        elif not is_none(mask_func):
            raise ValueError("Mask function for non-complex data should be None.")
        self.shift_mask = shift_mask
        self.mask_center_scale = mask_center_scale
        self.half_scan_percentage = half_scan_percentage
        self.remask = remask
        self.crop_size = crop_size
        if self.complex_data:
            self.kspace_crop = kspace_crop
        elif kspace_crop:
            raise ValueError("K-space crop for non-complex data should be None.")
        self.crop_before_masking = crop_before_masking
        if self.complex_data:
            self.kspace_zero_filling_size = kspace_zero_filling_size
        elif not is_none(kspace_zero_filling_size):
            raise ValueError("K-space zero filling size for non-complex data should be None.")
        self.normalize_inputs = normalize_inputs
        self.max_norm = max_norm
        self.fft_centered = fft_centered
        self.fft_normalization = fft_normalization
        self.spatial_dims = spatial_dims if spatial_dims is not None else [-2, -1]
        if self.complex_data:
            self.coil_dim = coil_dim - 1

        elif not is_none(coil_dim):
            raise ValueError("Coil dimension for non-complex data should be None.")
        if self.complex_data:
            self.apply_prewhitening = apply_prewhitening
            self.prewhitening = (
                NoisePreWhitening(
                    patch_size=[
                        prewhitening_patch_start,
                        prewhitening_patch_length + prewhitening_patch_start,
                        prewhitening_patch_start,
                        prewhitening_patch_length + prewhitening_patch_start,
                    ],
                    scale_factor=prewhitening_scale_factor,
                )
                if apply_prewhitening
                else None
            )

        elif apply_prewhitening:
            raise ValueError("Prewhitening for non-complex data cannot be applied.")
        if self.complex_data:
            self.gcc = (
                GeometricDecompositionCoilCompression(
                    virtual_coils=gcc_virtual_coils,
                    calib_lines=gcc_calib_lines,
                    align_data=gcc_align_data,
                    fft_centered=self.fft_centered,
                    fft_normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                )
                if apply_gcc
                else None
            )

        elif apply_gcc:
            raise ValueError("GCC for non-complex data cannot be applied.")
        self.use_seed = use_seed

    def __call__(
        self,
        kspace: np.ndarray,
        imspace: np.ndarray,
        sensitivity_map: np.ndarray,
        mask: np.ndarray,
        segmentation_labels: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_idx: int,
    ) -> Tuple[
        Tensor,
        Union[Union[Tensor, List[Union[Union[float, Tensor], Any]], float], Any],
        Union[Optional[Tensor], Any],
        Union[Union[List[Union[Tensor, Any]], Tensor, List[Tensor]], Any],
        Union[List[Tensor], Tensor],
        Union[Tensor, Any],
        Union[Optional[Tensor], Any],
        str,
        int,
        Union[List[int], int, Tensor],
    ]:
        """
        Apply the data transform.

        Parameters
        ----------
        kspace: The kspace.
        imspace: The image space.
        sensitivity_map: The sensitivity map.
        mask: List, sampling mask if exists and brain mask and head mask.
        segmentation_labels: The segmentation labels.
        attrs: The attributes.
        fname: The file name.
        slice_idx: The slice number.

        Returns
        -------
        The transformed data.
        """
        if not self.complex_data:
            imspace = torch.from_numpy(imspace)
        else:
            kspace = to_tensor(kspace)

            # This condition is necessary in case of auto estimation of sense maps.
            if sensitivity_map is not None and sensitivity_map.size != 0:
                sensitivity_map = to_tensor(sensitivity_map)

            if isinstance(mask, list):
                mask = [torch.from_numpy(m) for m in mask]
            elif mask.ndim != 0:
                mask = torch.from_numpy(mask)

        if segmentation_labels is not None and segmentation_labels.size != 0:
            segmentation_labels = torch.from_numpy(segmentation_labels)

        if self.complex_data:
            if self.apply_prewhitening:
                kspace = torch.stack(
                    [self.prewhitening(kspace[echo]) for echo in range(kspace.shape[0])], dim=0  # type: ignore
                )

            if self.gcc is not None:
                kspace = torch.stack([self.gcc(kspace[echo]) for echo in range(kspace.shape[0])], dim=0)
                if isinstance(sensitivity_map, torch.Tensor):
                    sensitivity_map = ifft2(
                        self.gcc(
                            fft2(
                                sensitivity_map,
                                centered=self.fft_centered,
                                normalization=self.fft_normalization,
                                spatial_dims=self.spatial_dims,
                            )
                        ),
                        centered=self.fft_centered,
                        normalization=self.fft_normalization,
                        spatial_dims=self.spatial_dims,
                    )

            # Apply zero-filling on kspace
            if self.kspace_zero_filling_size is not None and self.kspace_zero_filling_size not in ("", "None"):
                padding_top = np.floor_divide(abs(int(self.kspace_zero_filling_size[0]) - kspace.shape[2]), 2)
                padding_bottom = padding_top
                padding_left = np.floor_divide(abs(int(self.kspace_zero_filling_size[1]) - kspace.shape[3]), 2)
                padding_right = padding_left

                kspace = torch.view_as_complex(kspace)
                kspace = torch.nn.functional.pad(
                    kspace, pad=(padding_left, padding_right, padding_top, padding_bottom), mode="constant", value=0
                )
                kspace = torch.view_as_real(kspace)

                sensitivity_map = fft2(
                    sensitivity_map,
                    centered=self.fft_centered,
                    normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                )
                sensitivity_map = torch.view_as_complex(sensitivity_map)
                sensitivity_map = torch.nn.functional.pad(
                    sensitivity_map,
                    pad=(padding_left, padding_right, padding_top, padding_bottom),
                    mode="constant",
                    value=0,
                )
                sensitivity_map = torch.view_as_real(sensitivity_map)
                sensitivity_map = ifft2(
                    sensitivity_map,
                    centered=self.fft_centered,
                    normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                )

            # If the target is not given, we need to compute it.
            if self.coil_combination_method.upper() == "RSS":
                target_reconstruction = rss(
                    ifft2(
                        kspace,
                        centered=self.fft_centered,
                        normalization=self.fft_normalization,
                        spatial_dims=self.spatial_dims,
                    ),
                    dim=self.coil_dim,
                )
            elif self.coil_combination_method.upper() == "SENSE":
                if sensitivity_map is not None and sensitivity_map.size != 0:
                    target_reconstruction = sense(
                        ifft2(
                            kspace,
                            centered=self.fft_centered,
                            normalization=self.fft_normalization,
                            spatial_dims=self.spatial_dims,
                        ),
                        sensitivity_map,
                        dim=self.coil_dim,
                    )
            elif "target" in attrs or "target_rss" in attrs:
                target_reconstruction = torch.tensor(attrs["target"])
            else:
                raise ValueError("No target found")

            target_reconstruction = torch.view_as_complex(target_reconstruction)  # type: ignore
            target_reconstruction = torch.abs(target_reconstruction / torch.max(torch.abs(target_reconstruction)))
        else:
            target_reconstruction = imspace

        seed = tuple(map(ord, fname)) if self.use_seed else None
        acq_start = attrs["padding_left"] if "padding_left" in attrs else 0
        acq_end = attrs["padding_right"] if "padding_left" in attrs else 0

        # This should be outside the condition because it needs to be returned in the end, even if cropping is off.
        # crop_size = torch.tensor([attrs["recon_size"][0], attrs["recon_size"][1]])
        crop_size = target_reconstruction.shape[1:]
        if self.crop_size is not None and self.crop_size not in ("", "None"):
            # Check for smallest size against the target shape.
            h = min(int(self.crop_size[0]), target_reconstruction.shape[1])
            w = min(int(self.crop_size[1]), target_reconstruction.shape[2])

            # Check for smallest size against the stored recon shape in metadata.
            if crop_size[0] != 0:
                h = h if h <= crop_size[0] else crop_size[0]
            if crop_size[1] != 0:
                w = w if w <= crop_size[1] else crop_size[1]

            self.crop_size = (int(h), int(w))

            target_reconstruction = center_crop(target_reconstruction, self.crop_size)

            if self.complex_data:
                if sensitivity_map is not None and sensitivity_map.size != 0:
                    sensitivity_map = (
                        ifft2(
                            complex_center_crop(
                                fft2(
                                    sensitivity_map,
                                    centered=self.fft_centered,
                                    normalization=self.fft_normalization,
                                    spatial_dims=self.spatial_dims,
                                ),
                                self.crop_size,
                            ),
                            centered=self.fft_centered,
                            normalization=self.fft_normalization,
                            spatial_dims=self.spatial_dims,
                        )
                        if self.kspace_crop
                        else complex_center_crop(sensitivity_map, self.crop_size)
                    )

            if segmentation_labels is not None:
                segmentation_labels = center_crop(segmentation_labels, self.crop_size)

        if self.complex_data:
            # Cropping before masking will maintain the shape of original kspace intact for masking.
            if self.crop_size is not None and self.crop_size not in ("", "None") and self.crop_before_masking:
                kspace = (
                    complex_center_crop(kspace, self.crop_size)
                    if self.kspace_crop
                    else fft2(
                        complex_center_crop(
                            ifft2(
                                kspace,
                                centered=self.fft_centered,
                                normalization=self.fft_normalization,
                                spatial_dims=self.spatial_dims,
                            ),
                            self.crop_size,
                        ),
                        centered=self.fft_centered,
                        normalization=self.fft_normalization,
                        spatial_dims=self.spatial_dims,
                    )
                )

            if isinstance(mask, list):
                masked_kspaces = []
                masks = []
                for _mask in mask:
                    if list(_mask.shape) == [kspace.shape[-3], kspace.shape[-2]]:
                        _mask = _mask.unsqueeze(0).unsqueeze(-1)

                    padding = (acq_start, acq_end)
                    if (not is_none(padding[0]) and not is_none(padding[1])) and padding[0] != 0:
                        _mask[:, :, : padding[0]] = 0
                        _mask[:, :, padding[1] :] = 0  # padding value inclusive on right of zeros

                    if isinstance(_mask, np.ndarray):
                        _mask = torch.from_numpy(_mask).unsqueeze(0).unsqueeze(-1)

                    if self.shift_mask:
                        _mask = torch.fft.fftshift(_mask, dim=(self.spatial_dims[0] - 1, self.spatial_dims[1] - 1))

                    if self.crop_size is not None and self.crop_size not in ("", "None") and self.crop_before_masking:
                        _mask = complex_center_crop(_mask, self.crop_size)

                    masked_kspaces.append(kspace * _mask + 0.0)
                    masks.append(_mask)
                masked_kspace = masked_kspaces
                mask = masks
                acc = 1
            elif not is_none(mask) and mask.ndim != 0:  # and not is_none(self.mask_func):
                for _mask in mask:
                    if list(_mask.shape) == [kspace.shape[-3], kspace.shape[-2]]:
                        mask = torch.from_numpy(_mask).unsqueeze(0).unsqueeze(-1)
                        break

                padding = (acq_start, acq_end)
                if (not is_none(padding[0]) and not is_none(padding[1])) and padding[0] != 0:
                    mask[:, :, : padding[0]] = 0
                    mask[:, :, padding[1] :] = 0  # padding value inclusive on right of zeros

                if isinstance(mask, np.ndarray):
                    mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(-1)

                if self.shift_mask:
                    mask = torch.fft.fftshift(mask, dim=(self.spatial_dims[0] - 1, self.spatial_dims[1] - 1))

                if self.crop_size is not None and self.crop_size not in ("", "None") and self.crop_before_masking:
                    mask = complex_center_crop(mask, self.crop_size)

                masked_kspace = kspace * mask + 0.0  # the + 0.0 removes the sign of the zeros

                acc = 1
            elif is_none(self.mask_func):
                masked_kspace = kspace.clone()
                acc = torch.tensor([1])

                if mask is None or mask.ndim == 0:
                    mask = torch.ones(masked_kspace.shape[-3], masked_kspace.shape[-2]).type(torch.float32)  # type: ignore
                else:
                    mask = torch.from_numpy(mask)

                    if mask.dim() == 1:
                        mask = mask.unsqueeze(0)

                    if mask.shape[0] == masked_kspace.shape[2]:  # type: ignore
                        mask = mask.permute(1, 0)
                    elif mask.shape[0] != masked_kspace.shape[1]:  # type: ignore
                        mask = torch.ones(
                            [masked_kspace.shape[-3], masked_kspace.shape[-2]], dtype=torch.float32  # type: ignore
                        )

                if mask.shape[-2] == 1:  # 1D mask
                    mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(-1)
                else:  # 2D mask
                    # Crop loaded mask.
                    if self.crop_size is not None and self.crop_size not in ("", "None"):
                        mask = center_crop(mask, self.crop_size)
                    mask = mask.unsqueeze(0).unsqueeze(-1)

                if self.shift_mask:
                    mask = torch.fft.fftshift(mask, dim=(1, 2))

                masked_kspace = masked_kspace * mask
                mask = mask.byte()
            elif isinstance(self.mask_func, list):
                masked_kspaces = []
                masks = []
                accs = []
                for m in self.mask_func:
                    if self.dimensionality == 2:
                        _masked_kspace, _mask, _acc = apply_mask(
                            kspace,
                            m,
                            seed,
                            (acq_start, acq_end),
                            shift=self.shift_mask,
                            half_scan_percentage=self.half_scan_percentage,
                            center_scale=self.mask_center_scale,
                        )
                    elif self.dimensionality == 3:
                        _masked_kspace = []
                        _mask = None
                        for i in range(kspace.shape[0]):
                            _i_masked_kspace, _i_mask, _i_acc = apply_mask(
                                kspace[i],
                                m,
                                seed,
                                (acq_start, acq_end),
                                shift=self.shift_mask,
                                half_scan_percentage=self.half_scan_percentage,
                                center_scale=self.mask_center_scale,
                                existing_mask=_mask,
                            )
                            if self.remask:
                                _mask = _i_mask
                            if i == 0:
                                _acc = _i_acc
                            _masked_kspace.append(_i_masked_kspace)
                        _masked_kspace = torch.stack(_masked_kspace, dim=0)
                        _mask = _i_mask.unsqueeze(0)
                    else:
                        raise ValueError(f"Unsupported data dimensionality {self.dimensionality}D.")
                    masked_kspaces.append(_masked_kspace)
                    masks.append(_mask.byte())
                    accs.append(_acc)
                masked_kspace = masked_kspaces
                mask = masks
                acc = accs  # type: ignore
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

            # Cropping after masking.
            if self.crop_size is not None and self.crop_size not in ("", "None") and not self.crop_before_masking:
                kspace = (
                    complex_center_crop(kspace, self.crop_size)
                    if self.kspace_crop
                    else fft2(
                        complex_center_crop(
                            ifft2(
                                kspace,
                                centered=self.fft_centered,
                                normalization=self.fft_normalization,
                                spatial_dims=self.spatial_dims,
                            ),
                            self.crop_size,
                        ),
                        centered=self.fft_centered,
                        normalization=self.fft_normalization,
                        spatial_dims=self.spatial_dims,
                    )
                )

                masked_kspace = (
                    complex_center_crop(masked_kspace, self.crop_size)
                    if self.kspace_crop
                    else fft2(
                        complex_center_crop(
                            ifft2(
                                masked_kspace,
                                centered=self.fft_centered,
                                normalization=self.fft_normalization,
                                spatial_dims=self.spatial_dims,
                            ),
                            self.crop_size,
                        ),
                        centered=self.fft_centered,
                        normalization=self.fft_normalization,
                        spatial_dims=self.spatial_dims,
                    )
                )

                mask = center_crop(mask.squeeze(-1), self.crop_size).unsqueeze(-1)

            if self.normalize_inputs:
                if isinstance(self.mask_func, list):
                    if self.fft_normalization in ("backward", "ortho", "forward"):
                        imspace = ifft2(
                            kspace,
                            centered=self.fft_centered,
                            normalization=self.fft_normalization,
                            spatial_dims=self.spatial_dims,
                        )
                        if self.max_norm:
                            imspace = imspace / torch.max(torch.abs(imspace))
                        kspace = fft2(
                            imspace,
                            centered=self.fft_centered,
                            normalization=self.fft_normalization,
                            spatial_dims=self.spatial_dims,
                        )
                    elif self.fft_normalization in ("none", None) and self.max_norm:
                        imspace = torch.fft.ifftn(
                            torch.view_as_complex(kspace), dim=list(self.spatial_dims), norm=None
                        )
                        imspace = imspace / torch.max(torch.abs(imspace))
                        kspace = torch.view_as_real(torch.fft.fftn(imspace, dim=list(self.spatial_dims), norm=None))

                    masked_kspaces = []
                    for y in masked_kspace:
                        if self.fft_normalization in ("backward", "ortho", "forward"):
                            imspace = ifft2(
                                y,
                                centered=self.fft_centered,
                                normalization=self.fft_normalization,
                                spatial_dims=self.spatial_dims,
                            )
                            if self.max_norm:
                                imspace = imspace / torch.max(torch.abs(imspace))
                            y = fft2(
                                imspace,
                                centered=self.fft_centered,
                                normalization=self.fft_normalization,
                                spatial_dims=self.spatial_dims,
                            )
                        elif self.fft_normalization in ("none", None) and self.max_norm:
                            imspace = torch.fft.ifftn(torch.view_as_complex(y), dim=list(self.spatial_dims), norm=None)
                            imspace = imspace / torch.max(torch.abs(imspace))
                            y = torch.view_as_real(torch.fft.fftn(imspace, dim=list(self.spatial_dims), norm=None))
                        masked_kspaces.append(y)
                    masked_kspace = masked_kspaces
                elif self.fft_normalization in ("backward", "ortho", "forward"):
                    imspace = ifft2(
                        kspace,
                        centered=self.fft_centered,
                        normalization=self.fft_normalization,
                        spatial_dims=self.spatial_dims,
                    )
                    if self.max_norm:
                        imspace = imspace / torch.max(torch.abs(imspace))
                    kspace = fft2(
                        imspace,
                        centered=self.fft_centered,
                        normalization=self.fft_normalization,
                        spatial_dims=self.spatial_dims,
                    )
                    imspace = ifft2(
                        masked_kspace,
                        centered=self.fft_centered,
                        normalization=self.fft_normalization,
                        spatial_dims=self.spatial_dims,
                    )
                    if self.max_norm:
                        imspace = imspace / torch.max(torch.abs(imspace))
                    masked_kspace = fft2(
                        imspace,
                        centered=self.fft_centered,
                        normalization=self.fft_normalization,
                        spatial_dims=self.spatial_dims,
                    )
                elif self.fft_normalization in ("none", None) and self.max_norm:
                    imspace = torch.fft.ifftn(
                        torch.view_as_complex(masked_kspace), dim=list(self.spatial_dims), norm=None
                    )
                    imspace = imspace / torch.max(torch.abs(imspace))
                    masked_kspace = torch.view_as_real(torch.fft.fftn(imspace, dim=list(self.spatial_dims), norm=None))

                    imspace = torch.fft.ifftn(torch.view_as_complex(kspace), dim=list(self.spatial_dims), norm=None)
                    imspace = imspace / torch.max(torch.abs(imspace))
                    kspace = torch.view_as_real(torch.fft.fftn(imspace, dim=list(self.spatial_dims), norm=None))

                if self.max_norm:
                    if sensitivity_map.size != 0:
                        sensitivity_map = sensitivity_map / torch.max(torch.abs(sensitivity_map))
                    target_reconstruction = target_reconstruction / torch.max(torch.abs(target_reconstruction))

            if isinstance(self.mask_func, list):
                etas = []
                for y in masked_kspace:
                    if (
                        self.coil_combination_method.upper() == "SENSE"
                        and sensitivity_map is not None
                        and sensitivity_map.size != 0
                    ):
                        eta = sense(
                            ifft2(
                                y,
                                centered=self.fft_centered,
                                normalization=self.fft_normalization,
                                spatial_dims=self.spatial_dims,
                            ),
                            sensitivity_map,
                            dim=self.coil_dim,
                        )
                    else:
                        eta = rss(
                            ifft2(
                                y,
                                centered=self.fft_centered,
                                normalization=self.fft_normalization,
                                spatial_dims=self.spatial_dims,
                            ),
                            dim=self.coil_dim,
                        )
                    etas.append(eta)
                initial_prediction_reconstruction = etas
            else:
                if (
                    self.coil_combination_method.upper() == "SENSE"
                    and sensitivity_map is not None
                    and sensitivity_map.size != 0
                ):
                    initial_prediction_reconstruction = sense(
                        ifft2(
                            masked_kspace,
                            centered=self.fft_centered,
                            normalization=self.fft_normalization,
                            spatial_dims=self.spatial_dims,
                        ),
                        sensitivity_map,
                        dim=self.coil_dim,
                    )
                else:
                    initial_prediction_reconstruction = rss(
                        ifft2(
                            masked_kspace,
                            centered=self.fft_centered,
                            normalization=self.fft_normalization,
                            spatial_dims=self.spatial_dims,
                        ),
                        dim=self.coil_dim,
                    )
        else:
            if self.normalize_inputs:
                imspace = imspace / torch.max(torch.abs(imspace))

            kspace = torch.empty([])
            sensitivity_map = torch.empty([])
            masked_kspace = torch.empty([])
            initial_prediction_reconstruction = torch.abs(imspace)
            target_reconstruction = torch.empty([])
            acc = 1

        return (
            kspace,
            masked_kspace,
            sensitivity_map,
            mask,
            initial_prediction_reconstruction,
            target_reconstruction,
            torch.abs(segmentation_labels),
            fname,
            slice_idx,
            acc,
        )
