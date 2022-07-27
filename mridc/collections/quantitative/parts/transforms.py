# encoding: utf-8
__author__ = "Dimitrios Karkalousos, Chaoping Zhang"

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from skimage.restoration import unwrap_phase
from torch import Tensor
from torch.nn import functional as F

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

__all__ = ["qMRIDataTransforms"]


class qMRIDataTransforms:
    """qMRI preprocessing data transforms."""

    def __init__(
        self,
        TEs: Optional[List[float]],
        precompute_quantitative_maps: bool = True,
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
        fft_centered: bool = True,
        fft_normalization: str = "ortho",
        max_norm: bool = True,
        spatial_dims: Sequence[int] = None,
        coil_dim: int = 0,
        shift_B0_input: bool = False,
        use_seed: bool = True,
    ):
        """
        Initialize the data transform.

        Parameters
        ----------
        TEs: Echo times.
            List[float]
        precompute_quantitative_maps: Precompute quantitative maps.
            bool
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
        fft_centered: FFT centered.
            bool
        fft_normalization: FFT normalization.
            str
        max_norm: Normalization by the maximum value.
            bool
        spatial_dims: Spatial dimensions.
            Sequence[int]
        coil_dim: Coil dimension.
            int
        shift_B0_input: Shift B0 input.
            bool
        use_seed: Use seed.
            bool
        """
        self.TEs = TEs
        if self.TEs is None:
            raise ValueError("Please specify echo times (TEs).")
        self.precompute_quantitative_maps = precompute_quantitative_maps

        self.coil_combination_method = coil_combination_method
        self.dimensionality = dimensionality
        self.mask_func = mask_func
        self.shift_mask = shift_mask
        self.mask_center_scale = mask_center_scale
        self.half_scan_percentage = half_scan_percentage
        self.remask = remask
        self.crop_size = crop_size
        self.kspace_crop = kspace_crop
        self.crop_before_masking = crop_before_masking
        self.kspace_zero_filling_size = kspace_zero_filling_size
        self.normalize_inputs = normalize_inputs
        self.fft_centered = fft_centered
        self.fft_normalization = fft_normalization
        self.max_norm = max_norm
        self.spatial_dims = spatial_dims if spatial_dims is not None else [-2, -1]
        self.coil_dim = coil_dim - 1 if self.dimensionality == 2 else coil_dim

        self.shift_B0_input = shift_B0_input

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

        self.use_seed = use_seed

    def __call__(
        self,
        kspace: np.ndarray,
        sensitivity_map: np.ndarray,
        qmaps: np.ndarray,
        mask: np.ndarray,
        eta: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_idx: int,
    ) -> Tuple[
        Union[Tensor, List[Any], List[Tensor]],
        Union[Tensor, Any],
        Union[Tensor, List[Any], List[Tensor]],
        Union[Tensor, Any],
        Union[Tensor, List[Any], List[Tensor]],
        Union[Tensor, Any],
        Union[Tensor, List[Any], List[Tensor]],
        Union[Tensor, Any],
        Tensor,
        Tensor,
        Union[Union[Tensor, List[Union[Union[float, Tensor], Any]], float], Any],
        Union[Optional[Tensor], Any],
        Union[Union[List[Union[Tensor, Any]], Tensor, List[Tensor]], Any],
        Union[Tensor, Any],
        Union[Optional[Tensor], Any],
        Union[Tensor, Any],
        str,
        int,
        Union[List[int], int, Tensor],
    ]:
        """
        Apply the data transform.

        Parameters
        ----------
        kspace: The kspace.
        sensitivity_map: The sensitivity map.
        qmaps: The quantitative maps.
        mask: List, sampling mask if exists and brain mask and head mask.
        eta: The initial estimation.
        target: The target.
        attrs: The attributes.
        fname: The file name.
        slice_idx: The slice number.

        Returns
        -------
        The transformed data.
        """
        kspace = to_tensor(kspace)

        # This condition is necessary in case of auto estimation of sense maps.
        if sensitivity_map is not None and sensitivity_map.size != 0:
            sensitivity_map = to_tensor(sensitivity_map)

        mask_head = mask[2]
        mask_brain = mask[1]
        mask = mask[0]

        if mask_brain.ndim != 0:
            mask_brain = torch.from_numpy(mask_brain)

        if mask_head.ndim != 0:
            mask_head = torch.from_numpy(mask_head)

        if isinstance(mask, list):
            mask = [torch.from_numpy(m) for m in mask]
        elif mask.ndim != 0:
            mask = torch.from_numpy(mask)

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

        # Initial estimation
        eta = to_tensor(eta) if eta is not None and eta.size != 0 else torch.tensor([])

        # If the target is not given, we need to compute it.
        if self.coil_combination_method.upper() == "RSS":
            target = rss(
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
                target = sense(
                    ifft2(
                        kspace,
                        centered=self.fft_centered,
                        normalization=self.fft_normalization,
                        spatial_dims=self.spatial_dims,
                    ),
                    sensitivity_map.unsqueeze(0),
                    dim=self.coil_dim,
                )
        elif target is not None and target.size != 0:
            target = to_tensor(target)
        elif "target" in attrs or "target_rss" in attrs:
            target = torch.tensor(attrs["target"])
        else:
            raise ValueError("No target found")

        target = torch.view_as_complex(target)
        target = torch.abs(target / torch.max(torch.abs(target)))

        seed = tuple(map(ord, fname)) if self.use_seed else None
        acq_start = attrs["padding_left"] if "padding_left" in attrs else 0
        acq_end = attrs["padding_right"] if "padding_left" in attrs else 0

        # This should be outside the condition because it needs to be returned in the end, even if cropping is off.
        # crop_size = torch.tensor([attrs["recon_size"][0], attrs["recon_size"][1]])
        crop_size = target.shape[1:]
        if self.crop_size is not None and self.crop_size not in ("", "None"):
            # Check for smallest size against the target shape.
            h = min(int(self.crop_size[0]), target.shape[1])
            w = min(int(self.crop_size[1]), target.shape[2])

            # Check for smallest size against the stored recon shape in metadata.
            if crop_size[0] != 0:
                h = h if h <= crop_size[0] else crop_size[0]
            if crop_size[1] != 0:
                w = w if w <= crop_size[1] else crop_size[1]

            self.crop_size = (int(h), int(w))

            target = center_crop(target, self.crop_size)
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

            if eta is not None and eta.ndim > 2:
                eta = (
                    ifft2(
                        complex_center_crop(
                            fft2(
                                eta,
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
                    else complex_center_crop(eta, self.crop_size)
                )

            if mask_brain.dim() != 0:
                mask_brain = center_crop(mask_brain, self.crop_size)

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

            if mask is None:
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

        mask_head = torch.ones_like(mask_brain)

        if self.precompute_quantitative_maps:
            R2star_maps_init = []
            S0_maps_init = []
            B0_maps_init = []
            phi_maps_init = []
            etas = []
            for y in masked_kspace:
                eta = sense(
                    ifft2(
                        y,
                        centered=self.fft_centered,
                        normalization=self.fft_normalization,
                        spatial_dims=self.spatial_dims,
                    ),
                    sensitivity_map.unsqueeze(0),
                    dim=self.coil_dim,
                )
                etas.append(eta)
                R2star_map_init, S0_map_init, B0_map_init, phi_map_init = R2star_B0_real_S0_complex_mapping(
                    eta,
                    self.TEs,
                    mask_brain,
                    mask_head,
                    fully_sampled=True,
                    shift=self.shift_B0_input,
                    fft_centered=self.fft_centered,
                    fft_normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                )

                R2star_maps_init.append(R2star_map_init)
                S0_maps_init.append(S0_map_init)
                B0_maps_init.append(B0_map_init)
                phi_maps_init.append(phi_map_init)

            R2star_map_init = torch.stack(R2star_maps_init, dim=0)
            S0_map_init = torch.stack(S0_maps_init, dim=0)
            B0_map_init = torch.stack(B0_maps_init, dim=0)
            phi_map_init = torch.stack(phi_maps_init, dim=0)

            mask_brain_tmp = torch.ones_like(torch.abs(mask_brain))
            mask_brain_tmp = mask_brain_tmp.unsqueeze(0) if mask_brain.dim() == 2 else mask_brain_tmp
            imspace = sense(
                ifft2(
                    kspace,
                    centered=self.fft_centered,
                    normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                )
                * mask_brain_tmp.unsqueeze(self.coil_dim - 1).unsqueeze(-1),
                sensitivity_map.unsqueeze(0),
                dim=self.coil_dim,
            )

            R2star_map_target, S0_map_target, B0_map_target, phi_map_target = R2star_B0_real_S0_complex_mapping(
                imspace,
                self.TEs,
                mask_brain,
                mask_head,
                fully_sampled=True,
                shift=self.shift_B0_input,
                fft_centered=self.fft_centered,
                fft_normalization=self.fft_normalization,
                spatial_dims=self.spatial_dims,
            )
        else:
            if qmaps[0][0].ndim != 0:
                B0_map, S0_map, R2star_map, phi_map = qmaps
                B0_map = [torch.from_numpy(x).squeeze(0) for x in B0_map]
                B0_map_target = B0_map[-1]
                B0_map_init = B0_map[:-1]
                S0_map = [torch.from_numpy(x).squeeze(0) for x in S0_map]
                S0_map_target = S0_map[-1]
                S0_map_init = S0_map[:-1]
                R2star_map = [torch.from_numpy(x).squeeze(0) for x in R2star_map]
                R2star_map_target = R2star_map[-1]
                R2star_map_init = R2star_map[:-1]
                phi_map = [torch.from_numpy(x).squeeze(0) for x in phi_map]
                phi_map_target = phi_map[-1]
                phi_map_init = phi_map[:-1]
            else:
                B0_map_target = torch.tensor([])
                B0_map_init = [torch.tensor([])] * len(masked_kspace)
                S0_map_target = torch.tensor([])
                S0_map_init = [torch.tensor([])] * len(masked_kspace)
                R2star_map_target = torch.tensor([])
                R2star_map_init = [torch.tensor([])] * len(masked_kspace)
                phi_map_target = torch.tensor([])
                phi_map_init = [torch.tensor([])] * len(masked_kspace)

        # Normalize by the max value.
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
                    imspace = torch.fft.ifftn(torch.view_as_complex(kspace), dim=list(self.spatial_dims), norm=None)
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
                imspace = torch.fft.ifftn(torch.view_as_complex(masked_kspace), dim=list(self.spatial_dims), norm=None)
                imspace = imspace / torch.max(torch.abs(imspace))
                masked_kspace = torch.view_as_real(torch.fft.fftn(imspace, dim=list(self.spatial_dims), norm=None))

                imspace = torch.fft.ifftn(torch.view_as_complex(kspace), dim=list(self.spatial_dims), norm=None)
                imspace = imspace / torch.max(torch.abs(imspace))
                kspace = torch.view_as_real(torch.fft.fftn(imspace, dim=list(self.spatial_dims), norm=None))

            if self.max_norm:
                if sensitivity_map.size != 0:
                    sensitivity_map = sensitivity_map / torch.max(torch.abs(sensitivity_map))

                if eta.size != 0 and eta.ndim > 2:
                    eta = eta / torch.max(torch.abs(eta))

                target = target / torch.max(torch.abs(target))

        return (
            R2star_map_init,
            R2star_map_target,
            S0_map_init,
            S0_map_target,
            B0_map_init,
            B0_map_target,
            phi_map_init,
            phi_map_target,
            torch.tensor(self.TEs),
            kspace,
            masked_kspace,
            sensitivity_map,
            mask,
            mask_brain,
            eta,
            target,
            fname,
            slice_idx,
            acc,
        )


class GaussianSmoothing(torch.nn.Module):
    """
    Apply gaussian smoothing on a 1d, 2d or 3d tensor. Filtering is performed separately for each channel in the input
    using a depthwise convolution.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: Union[Optional[List[int]], int],
        sigma: float,
        dim: int = 2,
        shift: bool = False,
        fft_centered: bool = True,
        fft_normalization: str = "ortho",
        spatial_dims: Sequence[int] = None,
    ):
        """
        Initialize the module with the gaussian kernel size and standard deviation.

        Parameters
        ----------
        channels : int
            Number of channels in the input tensor.
        kernel_size : Union[Optional[List[int]], int]
            Gaussian kernel size.
        sigma : float
            Gaussian kernel standard deviation.
        dim : int
            Number of dimensions in the input tensor.
        shift : bool
            If True, the gaussian kernel is centered at (kernel_size - 1) / 2.
        fft_centered : bool
            Whether to center the FFT for a real- or complex-valued input.
        fft_normalization : str
            Whether to normalize the FFT output (None, "ortho", "backward", "forward", "none").
        spatial_dims : Sequence[int]
            Spatial dimensions to keep in the FFT.
        """
        super(GaussianSmoothing, self).__init__()

        self.shift = shift
        self.fft_centered = fft_centered
        self.fft_normalization = fft_normalization
        self.spatial_dims = spatial_dims

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * dim

        if isinstance(sigma, float):
            sigma = [sigma] * dim  # type: ignore

        # The gaussian kernel is the product of the gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size], indexing="ij"  # type: ignore
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):  # type: ignore
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-(((mgrid - mean) / std) ** 2) / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())  # type: ignore
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))  # type: ignore

        self.register_buffer("weight", kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError("Only 1, 2 and 3 dimensions are supported. Received {}.".format(dim))

    def forward(self, input):
        """
        Apply gaussian filter to input.

        Parameters
        ----------
        input : torch.Tensor
            Input to apply gaussian filter on.

        Returns
        -------
        torch.Tensor
            Filtered output.
        """
        if self.shift:
            input = input.permute(0, 2, 3, 1)
            input = ifft2(
                torch.fft.fftshift(
                    fft2(
                        torch.view_as_real(input[..., 0] + 1j * input[..., 1]),
                        self.fft_centered,
                        self.fft_normalization,
                        self.spatial_dims,
                    ),
                ),
                self.fft_centered,
                self.fft_normalization,
                self.spatial_dims,
            ).permute(0, 3, 1, 2)

        x = self.conv(input, weight=self.weight.to(input), groups=self.groups).to(input).detach()

        if self.shift:
            x = x.permute(0, 2, 3, 1)
            x = ifft2(
                torch.fft.fftshift(
                    fft2(
                        torch.view_as_real(x[..., 0] + 1j * x[..., 1]),
                        self.fft_centered,
                        self.fft_normalization,
                        self.spatial_dims,
                    ),
                ),
                self.fft_centered,
                self.fft_normalization,
                self.spatial_dims,
            ).permute(0, 3, 1, 2)

        return x


class LeastSquares:
    def __init__(self, device):
        super(LeastSquares, self).__init__()
        self.device = device

    def lstq(self, A, Y, lamb=0.0):
        """Differentiable least square."""
        q, r = torch.qr(A)
        return torch.inverse(r) @ q.permute(0, 2, 1) @ Y

    def lstq_pinv(self, A, Y, lamb=0.0):
        """Differentiable inverse least square."""
        if Y.dim() == 2:
            return torch.matmul(torch.pinverse(Y), A)
        else:
            return torch.matmul(
                torch.matmul(torch.inverse(torch.matmul(Y.permute(0, 2, 1), Y)), Y.permute(0, 2, 1)), A
            )

    def lstq_pinv_complex_np(self, A, Y, lamb=0.0):
        """Differentiable inverse least square for stacked complex inputs."""
        if Y.ndim == 2:
            return np.matmul(np.linalg.pinv(Y), A)
        else:
            Y = Y.to(self.device)
            A = A.to(Y)
            x = torch.matmul(torch.conj(Y).permute(0, 2, 1), Y)
            x = torch.matmul(torch.inverse(x), torch.conj(Y).permute(0, 2, 1))
            return torch.bmm(x, A)[..., 0]


def R2star_B0_real_S0_complex_mapping(
    prediction: torch.Tensor,
    TEs: Union[Optional[List[float]], float],
    brain_mask: torch.Tensor,
    head_mask: torch.Tensor,
    fully_sampled: bool = True,
    shift: bool = False,
    fft_centered: bool = True,
    fft_normalization: str = "ortho",
    spatial_dims: Sequence[int] = None,
):
    """
    Maps the prediction to R2*, B0, and S0 maps.

    Parameters
    ----------
    prediction : torch.Tensor
        The prediction of the model.
    TEs : Union[Optional[List[float]], float]
        The TEs of the images.
    brain_mask : torch.Tensor
        The brain mask of the images.
    head_mask : torch.Tensor
        The head mask of the images.
    fully_sampled : bool
        Whether the images are fully sampled.
    shift : bool
        If True, the gaussian kernel is centered at (kernel_size - 1) / 2.
    fft_centered : bool
        Whether to center the FFT for a real- or complex-valued input.
    fft_normalization : str
        Whether to normalize the FFT output (None, "ortho", "backward", "forward", "none").
    spatial_dims : Sequence[int]
        Spatial dimensions to keep in the FFT.

    Returns
    -------
    R2star : torch.Tensor
        The R2* map.
    B0 : torch.Tensor
        The B0 map.
    S0 : torch.Tensor
        The S0 map.
    phi : torch.Tensor
        The phi map.
    """
    R2star_map = R2star_S0_mapping(prediction, TEs)
    B0_map = -B0_phi_mapping(
        prediction,
        TEs,
        brain_mask,
        head_mask,
        fully_sampled,
        shift=shift,
        fft_centered=fft_centered,
        fft_normalization=fft_normalization,
        spatial_dims=spatial_dims,
    )[0]
    S0_map_real, S0_map_imag = S0_mapping_complex(
        prediction,
        TEs,
        R2star_map,
        B0_map,
        shift=shift,
        fft_centered=fft_centered,
        fft_normalization=fft_normalization,
        spatial_dims=spatial_dims,
    )
    return R2star_map, S0_map_real, B0_map, S0_map_imag


def R2star_S0_mapping(
    prediction: torch.Tensor,
    TEs: Union[Optional[List[float]], float],
    scaling_factor: float = 1e-3,
):
    """
    R2* map and S0 map estimation for multi-echo GRE from stored magnitude image files acquired at multiple TEs.

    Parameters
    ----------
    prediction : torch.Tensor
        The prediction of the model.
    TEs : Union[Optional[List[float]], float]
        The TEs of the images.
    scaling_factor : float
        The scaling factor.

    Returns
    -------
    R2star : torch.Tensor
        The R2* map.
    S0 : torch.Tensor
        The S0 map.
    """
    prediction = torch.abs(torch.view_as_complex(prediction)) + 1e-8
    prediction_flatten = torch.flatten(prediction, start_dim=1, end_dim=-1).detach().cpu()  # .numpy()
    TEs = np.array(TEs).to(prediction_flatten)

    # TODO: this part needs a proper implementation in PyTorch
    R2star_map = torch.zeros([prediction_flatten.shape[1]])
    for i in range(prediction_flatten.shape[1]):
        R2star_map[i], _ = torch.from_numpy(
            np.polyfit(
                TEs * scaling_factor,  # type:ignore
                np.log(prediction_flatten[:, i]),
                1,
                w=np.sqrt(prediction_flatten[:, i]),
            )
        ).to(prediction)
    R2star_map = torch.reshape(-R2star_map, prediction.shape[1:4])
    return R2star_map


def B0_phi_mapping(
    prediction: torch.Tensor,
    TEs: Union[Optional[List[float]], float],
    brain_mask: torch.Tensor,
    head_mask: torch.Tensor,
    fully_sampled: bool = True,
    scaling_factor: float = 1e-3,
    shift: bool = False,
    fft_centered: bool = True,
    fft_normalization: str = "ortho",
    spatial_dims: Sequence[int] = None,
):
    """
    B0 map and Phi map estimation for multi-echo GRE from stored magnitude image files acquired at multiple TEs.

    Parameters
    ----------
    prediction : torch.Tensor
        The prediction of the model.
    TEs : Union[Optional[List[float]], float]
        The TEs of the images.
    brain_mask : torch.Tensor
        The brain mask of the images.
    head_mask : torch.Tensor
        The head mask of the images.
    fully_sampled : bool
        Whether the images are fully sampled.
    scaling_factor : float
        The scaling factor.
    shift : bool
        If True, the gaussian kernel is centered at (kernel_size - 1) / 2.
    fft_centered : bool
        Whether to center the FFT for a real- or complex-valued input.
    fft_normalization : str
        Whether to normalize the FFT output (None, "ortho", "backward", "forward", "none").
    spatial_dims : Sequence[int]
        Spatial dimensions to keep in the FFT.

    Returns
    -------
    B0 : torch.Tensor
        The B0 map.
    phi : torch.Tensor
        The phi map.
    """
    lsq = LeastSquares(device=prediction.device)

    TEnotused = 3  # if fully_sampled else 3
    TEs = torch.tensor(TEs)

    # brain_mask is used only for descale of phase difference (so that phase_diff is in between -2pi and 2pi)
    brain_mask_descale = brain_mask
    shape = prediction.shape

    # apply gaussian blur with radius r to
    smoothing = GaussianSmoothing(
        channels=2,
        kernel_size=9,
        sigma=1.0,
        dim=2,
        shift=shift,
        fft_centered=fft_centered,
        fft_normalization=fft_normalization,
        spatial_dims=spatial_dims,
    )
    prediction = prediction.unsqueeze(1).permute([0, 1, 4, 2, 3])  # add a dummy batch dimension
    for i in range(prediction.shape[0]):
        prediction[i] = smoothing(F.pad(prediction[i], (4, 4, 4, 4), mode="reflect"))
    prediction = prediction.permute([0, 1, 3, 4, 2]).squeeze(1)

    if shift:
        prediction = ifft2(
            torch.fft.fftshift(fft2(prediction, fft_centered, fft_normalization, spatial_dims), dim=(1, 2)),
            fft_centered,
            fft_normalization,
            spatial_dims,
        )

    phase = torch.angle(torch.view_as_complex(prediction))

    # unwrap phases
    phase_unwrapped = torch.zeros_like(phase)
    mask_head_np = np.invert(head_mask.cpu().detach().numpy() > 0.5)

    # loop over echo times
    for i in range(phase.shape[0]):
        phase_unwrapped[i] = torch.from_numpy(
            unwrap_phase(np.ma.array(phase[i].detach().cpu().numpy(), mask=mask_head_np)).data
        ).to(prediction)

    phase_diff_set = []
    TE_diff = []

    # obtain phase differences and TE differences
    for i in range(phase_unwrapped.shape[0] - TEnotused):
        phase_diff_set.append(torch.flatten(phase_unwrapped[i + 1] - phase_unwrapped[i]))
        phase_diff_set[i] = (
            phase_diff_set[i]
            - torch.round(
                torch.abs(
                    torch.sum(phase_diff_set[i] * torch.flatten(brain_mask_descale))
                    / torch.sum(brain_mask_descale)
                    / 2
                    / np.pi
                )
            )
            * 2
            * np.pi
        )
        TE_diff.append(TEs[i + 1] - TEs[i])  # type: ignore

    phase_diff_set = torch.stack(phase_diff_set, 0)
    TE_diff = torch.stack(TE_diff, 0).to(prediction)

    # least squares fitting to obtain phase map
    B0_map_tmp = lsq.lstq_pinv(
        phase_diff_set.unsqueeze(2).permute(1, 0, 2), TE_diff.unsqueeze(1) * scaling_factor  # type: ignore
    )
    B0_map = B0_map_tmp.reshape(shape[-3], shape[-2])
    B0_map = B0_map * torch.abs(head_mask)

    # obtain phi map
    phi_map = (phase_unwrapped[0] - scaling_factor * TEs[0] * B0_map).squeeze(0)  # type: ignore

    return B0_map.to(prediction), phi_map.to(prediction)


def S0_mapping_complex(
    prediction: torch.Tensor,
    TEs: Union[Optional[List[float]], float],
    R2star_map: torch.Tensor,
    B0_map: torch.Tensor,
    scaling_factor: float = 1e-3,
    shift: bool = False,
    fft_centered: bool = True,
    fft_normalization: str = "ortho",
    spatial_dims: Sequence[int] = None,
):
    """
    Complex S0 mapping.

    Parameters
    ----------
    prediction : torch.Tensor
        The prediction of the model.
    TEs : Union[Optional[List[float]], float]
        The TEs of the images.
    R2star_map : torch.Tensor
        The R2* map.
    B0_map : torch.Tensor
        The B0 map.
    scaling_factor : float
        The scaling factor.
    shift : bool
        If True, the gaussian kernel is centered at (kernel_size - 1) / 2.
    fft_centered : bool
        Whether to center the FFT for a real- or complex-valued input.
    fft_normalization : str
        Whether to normalize the FFT output (None, "ortho", "backward", "forward", "none").
    spatial_dims : Sequence[int]
        Spatial dimensions to keep in the FFT.

    Returns
    -------
    S0 : torch.Tensor
        The S0 map.
    """
    lsq = LeastSquares(device=prediction.device)

    prediction = torch.view_as_complex(prediction)
    prediction_flatten = prediction.reshape(prediction.shape[0], -1)

    TEs = torch.tensor(TEs).to(prediction)

    R2star_B0_complex_map = R2star_map.to(prediction) + 1j * B0_map.to(prediction)
    R2star_B0_complex_map_flatten = R2star_B0_complex_map.flatten()

    TEs_r2 = TEs[0:4].unsqueeze(1) * -R2star_B0_complex_map_flatten  # type: ignore

    S0_map = lsq.lstq_pinv_complex_np(
        prediction_flatten.permute(1, 0).unsqueeze(2),
        torch.exp(scaling_factor * TEs_r2.permute(1, 0).unsqueeze(2)),
    )

    S0_map = torch.view_as_real(S0_map.reshape(prediction.shape[1:]))

    if shift:
        S0_map = ifft2(
            torch.fft.fftshift(fft2(S0_map, fft_centered, fft_normalization, spatial_dims), dim=(0, 1)),
            fft_centered,
            fft_normalization,
            spatial_dims,
        )

    S0_map_real, S0_map_imag = torch.chunk(S0_map, 2, dim=-1)

    return S0_map_real.squeeze(-1), S0_map_imag.squeeze(-1)
