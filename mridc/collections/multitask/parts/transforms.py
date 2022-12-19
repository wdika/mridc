# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor

import mridc.collections.common.parts.fft as fft
import mridc.collections.common.parts.utils as utils
import mridc.collections.reconstruction.data.subsample as subsample
import mridc.collections.reconstruction.parts.transforms as reconstruction_transforms

__all__ = ["MTLMRIDataTransforms"]


class MTLMRIDataTransforms:
    """MultiTask Learning MRI preprocessing data transforms."""

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
        mask_func: Optional[List[subsample.MaskFunc]] = None,
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
        echo_type: str = "echo1-echo2-mc",
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
            List[subsample.MaskFunc]
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
        echo_type: Echo type.
            str
        consecutive_slices: Number of consecutive slices.
            int
        use_seed: Use seed.
            bool
        """
        self.complex_data = complex_data
        if not self.complex_data:
            if not utils.is_none(coil_combination_method):
                raise ValueError("Coil combination method for non-complex data should be None.")
        else:
            self.coil_combination_method = coil_combination_method
        self.dimensionality = dimensionality
        if not self.complex_data:
            if not utils.is_none(mask_func):
                raise ValueError("Mask function for non-complex data should be None.")
        else:
            self.mask_func = mask_func
        self.shift_mask = shift_mask
        self.mask_center_scale = mask_center_scale
        self.half_scan_percentage = half_scan_percentage
        self.remask = remask
        self.crop_size = crop_size
        if not self.complex_data:
            if kspace_crop:
                raise ValueError("K-space crop for non-complex data should be None.")
        else:
            self.kspace_crop = kspace_crop
        self.crop_before_masking = crop_before_masking
        if not self.complex_data:
            if not utils.is_none(kspace_zero_filling_size):
                raise ValueError("K-space zero filling size for non-complex data should be None.")
        else:
            self.kspace_zero_filling_size = kspace_zero_filling_size
        self.normalize_inputs = normalize_inputs
        self.max_norm = max_norm
        self.fft_centered = fft_centered
        self.fft_normalization = fft_normalization
        self.spatial_dims = spatial_dims if spatial_dims is not None else [-2, -1]
        if not self.complex_data:
            if not utils.is_none(coil_dim):
                raise ValueError("Coil dimension for non-complex data should be None.")
        else:
            self.coil_dim = coil_dim - 1

        self.echo_type = echo_type

        if not self.complex_data:
            if apply_prewhitening:
                raise ValueError("Prewhitening for non-complex data cannot be applied.")
        else:
            self.apply_prewhitening = apply_prewhitening
            self.prewhitening = (
                reconstruction_transforms.NoisePreWhitening(
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

        if not self.complex_data:
            if apply_gcc:
                raise ValueError("GCC for non-complex data cannot be applied.")
        else:
            self.gcc = (
                reconstruction_transforms.GeometricDecompositionCoilCompression(
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
        imspace: np.ndarray,
        masked_kspace: np.ndarray,
        sensitivity_map: np.ndarray,
        mask: np.ndarray,
        segmentation_labels: np.ndarray,
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
        masked_kspace: The masked kspace.
        sensitivity_map: The sensitivity map.
        mask: List, sampling mask if exists and brain mask and head mask.
        segmentation_labels: The segmentation labels.
        fname: The file name.
        slice_idx: The slice number.

        Returns
        -------
        The transformed data.
        """
        if not self.complex_data:
            imspace = torch.from_numpy(imspace)
            target_reconstruction = imspace
        else:
            kspace = utils.to_tensor(kspace)

            if masked_kspace is not None and masked_kspace.ndim != 0:
                masked_kspace = utils.to_tensor(masked_kspace)

            # This condition is necessary in case of auto estimation of sense maps.
            if sensitivity_map is not None and sensitivity_map.ndim != 0:
                sensitivity_map = utils.to_tensor(sensitivity_map)

            if self.echo_type == "echo1":
                kspace = kspace[:, :, 0, :, :]
                masked_kspace = masked_kspace[:, :, 0, :, :]
                sensitivity_map = sensitivity_map[:, :, 0, :, :]
            elif self.echo_type == "echo2":
                kspace = kspace[:, :, 1, :, :]
                masked_kspace = masked_kspace[:, :, 1, :, :]
                sensitivity_map = sensitivity_map[:, :, 0, :, :]
            elif self.echo_type == "echo1+echo2":
                kspace = kspace[:, :, 0, :, :] + kspace[:, :, 1, :, :]
                masked_kspace = masked_kspace[:, :, 0, :, :] + masked_kspace[:, :, 1, :, :]
                sensitivity_map = sensitivity_map[:, :, 0, :, :]
            elif self.echo_type == "echo1-echo2-mc":
                kspace = torch.cat((kspace[:, :, 0, :, :], kspace[:, :, 1, :, :]), dim=self.coil_dim)
                masked_kspace = torch.cat((masked_kspace[:, :, 0, :, :], masked_kspace[:, :, 1, :, :]), dim=self.coil_dim)
                sensitivity_map = torch.cat((sensitivity_map[:, :, 0, :, :], sensitivity_map[:, :, 0, :, :]), dim=self.coil_dim)
            else:
                raise ValueError(f"Invalid echo type {self.echo_type}.")

            # permute coil dimension to the first dimension
            kspace = kspace.permute(2, 0, 1, 3)
            masked_kspace = masked_kspace.permute(2, 0, 1, 3)
            sensitivity_map = sensitivity_map.permute(2, 0, 1, 3)

            mask = torch.ones(
                masked_kspace.shape[1], masked_kspace.shape[2]
            ).unsqueeze(0).unsqueeze(-1).to(masked_kspace.device)

            if self.apply_prewhitening:
                kspace = torch.stack(
                    [self.prewhitening(kspace[echo]) for echo in range(kspace.shape[0])],  # type: ignore
                    dim=0,
                )

            if self.gcc is not None:
                kspace = torch.stack([self.gcc(kspace[echo]) for echo in range(kspace.shape[0])], dim=0)
                if isinstance(sensitivity_map, torch.Tensor):
                    sensitivity_map = fft.ifft2(
                        self.gcc(
                            fft.fft2(
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

                sensitivity_map = fft.fft2(
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
                sensitivity_map = fft.ifft2(
                    sensitivity_map,
                    centered=self.fft_centered,
                    normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                )

            # If the target is not given, we need to compute it.
            imspace = fft.ifft2(
                kspace,
                centered=self.fft_centered,
                normalization=self.fft_normalization,
                spatial_dims=self.spatial_dims
            )
            target_reconstruction = torch.view_as_complex(utils.coil_combination(
                imspace, sensitivity_map, self.coil_combination_method.upper(), dim=self.coil_dim
            ))  # type: ignore

            masked_imspace = fft.ifft2(
                masked_kspace * mask, centered=self.fft_centered,
                normalization=self.fft_normalization, spatial_dims=self.spatial_dims
            )
            initial_prediction = utils.coil_combination(
                masked_imspace, sensitivity_map, self.coil_combination_method.upper(), dim=self.coil_dim
            )

        if segmentation_labels is not None and segmentation_labels.ndim > 1:
            segmentation_labels = torch.from_numpy(segmentation_labels)
        else:
            segmentation_labels = torch.empty([])

        seed = tuple(map(ord, fname)) if self.use_seed else None

        # This should be outside the condition because it needs to be returned in the end, even if cropping is off.
        # crop_size = torch.tensor([attrs["recon_size"][0], attrs["recon_size"][1]])
        crop_size = target_reconstruction.shape[:2] if target_reconstruction.dim() > 2 else target_reconstruction.shape
        if self.crop_size is not None and self.crop_size not in ("", "None"):
            # Check for smallest size against the target shape.
            h = min(int(self.crop_size[0]), target_reconstruction.shape[0])
            w = min(int(self.crop_size[1]), target_reconstruction.shape[1])

            # Check for smallest size against the stored recon shape in metadata.
            if crop_size[0] != 0:
                h = h if h <= crop_size[0] else crop_size[0]
            if crop_size[1] != 0:
                w = w if w <= crop_size[1] else crop_size[1]

            self.crop_size = (int(h), int(w))

            target_reconstruction = utils.center_crop(target_reconstruction, self.crop_size)

            if self.complex_data and sensitivity_map is not None and sensitivity_map.dim() != 0:
                sensitivity_map = (
                    fft.ifft2(
                        utils.complex_center_crop(
                            fft.fft2(
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
                    else utils.complex_center_crop(sensitivity_map, self.crop_size)
                )

            if initial_prediction is not None and initial_prediction.dim() != 0:
                initial_prediction = (
                    fft.ifft2(
                        utils.complex_center_crop(
                            fft.fft2(
                                initial_prediction,
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
                    else utils.complex_center_crop(initial_prediction, self.crop_size)
                )

            if segmentation_labels is not None and segmentation_labels.dim() > 1:
                segmentation_labels = utils.center_crop(segmentation_labels, self.crop_size)

        if self.complex_data:
            # Cropping before masking will maintain the shape of original kspace intact for masking.
            if self.crop_size is not None and self.crop_size not in ("", "None") and self.crop_before_masking:
                kspace = (
                    utils.complex_center_crop(kspace, self.crop_size)
                    if self.kspace_crop
                    else fft.fft2(
                        utils.complex_center_crop(
                            fft.ifft2(
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

            acc = 1
            masked_kspace = masked_kspace * mask + 0.0  # the + 0.0 removes the sign of the zeros
            mask = mask.byte()

            # Cropping after masking.
            if self.crop_size is not None and self.crop_size not in ("", "None") and not self.crop_before_masking:
                kspace = (
                    utils.complex_center_crop(kspace, self.crop_size)
                    if self.kspace_crop
                    else fft.fft2(
                        utils.complex_center_crop(
                            fft.ifft2(
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
                    utils.complex_center_crop(masked_kspace, self.crop_size)
                    if self.kspace_crop
                    else fft.fft2(
                        utils.complex_center_crop(
                            fft.ifft2(
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

                mask = utils.center_crop(mask.squeeze(-1), self.crop_size).unsqueeze(-1)

            if self.normalize_inputs:
                if self.fft_normalization in ("backward", "ortho", "forward"):
                    imspace = fft.ifft2(
                        kspace,
                        centered=self.fft_centered,
                        normalization=self.fft_normalization,
                        spatial_dims=self.spatial_dims,
                    )
                    if self.max_norm:
                        imspace = imspace / torch.max(torch.abs(imspace))
                    kspace = fft.fft2(
                        imspace,
                        centered=self.fft_centered,
                        normalization=self.fft_normalization,
                        spatial_dims=self.spatial_dims,
                    )
                    masked_imspace = fft.ifft2(
                        masked_kspace,
                        centered=self.fft_centered,
                        normalization=self.fft_normalization,
                        spatial_dims=self.spatial_dims,
                    )
                    if self.max_norm:
                        masked_imspace = masked_imspace / torch.max(torch.abs(masked_imspace))
                    masked_kspace = fft.fft2(
                        masked_imspace,
                        centered=self.fft_centered,
                        normalization=self.fft_normalization,
                        spatial_dims=self.spatial_dims,
                    )
                elif self.fft_normalization in ("none", None) and self.max_norm:
                    masked_imspace = torch.fft.ifftn(
                        torch.view_as_complex(masked_kspace), dim=list(self.spatial_dims), norm=None
                    )
                    masked_imspace = masked_imspace / torch.max(torch.abs(masked_imspace))
                    masked_kspace = torch.view_as_real(
                        torch.fft.fftn(masked_imspace, dim=list(self.spatial_dims), norm=None)
                    )

                    imspace = torch.fft.ifftn(torch.view_as_complex(kspace), dim=list(self.spatial_dims), norm=None)
                    imspace = imspace / torch.max(torch.abs(imspace))
                    kspace = torch.view_as_real(torch.fft.fftn(imspace, dim=list(self.spatial_dims), norm=None))

                if self.max_norm:
                    if sensitivity_map.dim() != 0:
                        sensitivity_map = sensitivity_map / torch.max(torch.abs(sensitivity_map))
                    target_reconstruction = target_reconstruction / torch.max(torch.abs(target_reconstruction))

            initial_prediction_reconstruction = initial_prediction
            initial_prediction_reconstruction = initial_prediction_reconstruction / torch.max(
                torch.abs(initial_prediction_reconstruction)
            )
        else:
            if self.normalize_inputs:
                imspace = imspace / torch.max(torch.abs(imspace))

            kspace = torch.empty([])
            sensitivity_map = torch.empty([])
            masked_kspace = torch.empty([])
            initial_prediction_reconstruction = torch.abs(imspace)

            if target_reconstruction is None or target_reconstruction.dim() < 2:
                target_reconstruction = torch.empty([])
            else:
                if self.max_norm:
                    target_reconstruction = target_reconstruction / torch.max(torch.abs(target_reconstruction))

            acc = 1

        segmentation_labels = torch.abs(segmentation_labels)

        return (
            kspace,
            masked_kspace,
            sensitivity_map,
            mask,
            initial_prediction_reconstruction,
            target_reconstruction,
            segmentation_labels,
            fname,
            slice_idx,
            acc,
        )
