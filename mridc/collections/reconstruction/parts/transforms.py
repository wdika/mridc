# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

import mridc.collections.common.data.subsample as subsample
import mridc.collections.common.parts.fft as fft
import mridc.collections.common.parts.utils as utils
from mridc.collections.common.parts.transforms import (
    Cropper,
    GeometricDecompositionCoilCompression,
    Masker,
    NoisePreWhitening,
    Normalizer,
    ZeroFilling,
)

__all__ = ["ReconstructionMRIDataTransforms"]


class ReconstructionMRIDataTransforms:
    """
    Data transforms for accelerated-MRI reconstruction.

    Parameters
    ----------
    apply_prewhitening : bool, optional
        Apply prewhitening. If ``True`` then the prewhitening arguments are used. Default is ``False``.
    prewhitening_scale_factor : float, optional
        Prewhitening scale factor. Default is ``1.0``.
    prewhitening_patch_start : int, optional
        Prewhitening patch start. Default is ``10``.
    prewhitening_patch_length : int, optional
        Prewhitening patch length. Default is ``30``.
    apply_gcc : bool, optional
        Apply Geometric Decomposition Coil Compression. If ``True`` then the GCC arguments are used.
        Default is ``False``.
    gcc_virtual_coils : int, optional
        GCC virtual coils. Default is ``10``.
    gcc_calib_lines : int, optional
        GCC calibration lines. Default is ``24``.
    gcc_align_data : bool, optional
        GCC align data. Default is ``True``.
    coil_combination_method : str, optional
        Coil combination method. Default is ``"SENSE"``.
    dimensionality : int, optional
        Dimensionality. Default is ``2``.
    mask_func : Optional[List[subsample.MaskFunc]], optional
        Mask function to retrospectively undersample the k-space. Default is ``None``.
    shift_mask : bool, optional
        Whether to shift the mask. This needs to be set alongside with the ``fft_centered`` argument.
        Default is ``False``.
    mask_center_scale : Optional[float], optional
        Center scale of the mask. This defines how much densely sampled will be the center of k-space.
        Default is ``0.02``.
    half_scan_percentage : float, optional
        Whether to simulate a half scan. Default is ``0.0``.
    remask : bool, optional
        Use the same mask. Default is ``False``.
    crop_size : Optional[Tuple[int, int]], optional
        Center crop size. It applies cropping in image space. Default is ``None``.
    kspace_crop : bool, optional
        Whether to crop in k-space. Default is ``False``.
    crop_before_masking : bool, optional
        Whether to crop before masking. Default is ``True``.
    kspace_zero_filling_size : Optional[Tuple], optional
        Whether to apply zero filling in k-space. Default is ``None``.
    normalize_inputs : bool, optional
        Whether to normalize the inputs. Default is ``True``.
    normalization_type : str, optional
        Normalization type. Can be ``max`` or ``mean`` or ``minmax``. Default is ``max``.
    fft_centered : bool, optional
        Whether to center the FFT. Default is ``False``.
    fft_normalization : str, optional
        FFT normalization. Default is ``"backward"``.
    spatial_dims : Sequence[int], optional
        Spatial dimensions. Default is ``None``.
    coil_dim : int, optional
        Coil dimension. Default is ``0``, meaning that the coil dimension is the first dimension before applying batch.
    consecutive_slices : int, optional
        Consecutive slices. Default is ``1``.
    use_seed : bool, optional
        Whether to use seed. Default is ``True``.

    Returns
    -------
    ReconstructionMRIDataTransforms
        Data transformed for accelerated-MRI reconstruction.
    """

    def __init__(
        self,
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
        normalize_inputs: bool = True,
        normalization_type: str = "max",
        fft_centered: bool = False,
        fft_normalization: str = "backward",
        spatial_dims: Sequence[int] = None,
        coil_dim: int = 0,
        consecutive_slices: int = 1,
        use_seed: bool = True,
    ):
        super().__init__()

        self.coil_combination_method = coil_combination_method
        self.kspace_crop = kspace_crop
        self.crop_before_masking = crop_before_masking
        self.normalize_inputs = normalize_inputs

        self.fft_centered = fft_centered
        self.fft_normalization = fft_normalization
        self.spatial_dims = spatial_dims if spatial_dims is not None else [-2, -1]
        self.coil_dim = coil_dim - 1 if dimensionality == 2 else coil_dim

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

        self.kspace_zero_filling = (
            ZeroFilling(
                zero_filling_size=kspace_zero_filling_size,  # type: ignore
                spatial_dims=self.spatial_dims,  # type: ignore
            )
            if not utils.is_none(kspace_zero_filling_size)
            else None
        )

        self.masking = Masker(
            mask_func=mask_func,  # type: ignore
            spatial_dims=self.spatial_dims,  # type: ignore
            shift_mask=shift_mask,
            half_scan_percentage=half_scan_percentage,
            center_scale=mask_center_scale,  # type: ignore
            dimensionality=dimensionality,
            remask=remask,
        )

        self.cropping = (
            Cropper(
                cropping_size=crop_size,  # type: ignore
                spatial_dims=self.spatial_dims,  # type: ignore
            )
            if not utils.is_none(crop_size)
            else None
        )

        self.normalization = Normalizer(
            normalization_type=normalization_type,
            fft_centered=self.fft_centered,
            fft_normalization=self.fft_normalization,
            spatial_dims=self.spatial_dims,  # type: ignore
        )

        self.use_seed = use_seed

    def __call__(
        self,
        kspace: np.ndarray,
        sensitivity_map: np.ndarray,
        mask: np.ndarray,
        prediction: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_idx: int,
    ) -> Tuple[
        torch.Tensor,
        Union[Union[List, torch.Tensor], torch.Tensor],
        Union[Optional[torch.Tensor], Any],
        Union[List, Any],
        Union[Optional[torch.Tensor], Any],
        Union[torch.Tensor, Any],
        str,
        int,
        Union[List, Any],
    ]:
        """
        Apply the data transform.

        Parameters
        ----------
        kspace: The kspace.
        sensitivity_map: The sensitivity map.
        mask: The mask.
        prediction: The initial estimation.
        target: The target.
        attrs: The attributes.
        fname: The file name.
        slice_idx: The slice number.

        Returns
        -------
        The transformed data.
        """
        kspace = utils.to_tensor(kspace)

        kspace = utils.add_coil_dim_if_singlecoil(kspace, dim=self.coil_dim)

        # This condition is necessary in case of auto estimation of sense maps.
        if sensitivity_map is not None and sensitivity_map.size != 0:
            sensitivity_map = utils.to_tensor(sensitivity_map)
        else:
            # If no sensitivity map is provided, either the data is singlecoil or the sense net is used.
            # Initialize the sensitivity map to 1 to assure for the singlecoil case.
            sensitivity_map = torch.ones_like(kspace)

        if self.prewhitening is not None:
            kspace = self.prewhitening(kspace)  # type: ignore

        if self.gcc is not None:
            kspace = self.gcc(kspace)
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
        if self.kspace_zero_filling is not None:
            kspace = self.kspace_zero_filling(kspace)
            sensitivity_map = fft.fft2(
                sensitivity_map,
                centered=self.fft_centered,
                normalization=self.fft_normalization,
                spatial_dims=self.spatial_dims,
            )
            sensitivity_map = self.kspace_zero_filling(sensitivity_map)
            sensitivity_map = fft.ifft2(
                sensitivity_map,
                centered=self.fft_centered,
                normalization=self.fft_normalization,
                spatial_dims=self.spatial_dims,
            )

        # Initial estimation
        prediction = (
            utils.to_tensor(prediction) if prediction is not None and prediction.size != 0 else torch.tensor([])
        )

        if target is not None and target.ndim > 1:
            target = utils.to_tensor(target)
        elif not utils.is_none(self.coil_combination_method.upper()):
            if sensitivity_map is not None and sensitivity_map.size != 0:
                target = utils.coil_combination_method(
                    fft.ifft2(
                        kspace,
                        centered=self.fft_centered,
                        normalization=self.fft_normalization,
                        spatial_dims=self.spatial_dims,
                    ),
                    sensitivity_map,
                    method=self.coil_combination_method.upper(),
                    dim=self.coil_dim,
                )
        else:
            raise ValueError("No target found, while coil combination method is not defined.")
        if target.shape[-1] == 2:
            target = torch.view_as_complex(target)

        seed = tuple(map(ord, fname)) if self.use_seed else None

        acq_start = attrs["padding_left"] if "padding_left" in attrs else 0
        acq_end = attrs["padding_right"] if "padding_left" in attrs else 0

        if self.cropping is not None:
            target = self.cropping(target)
            if sensitivity_map is not None and sensitivity_map.size != 0:
                sensitivity_map = self.cropping(sensitivity_map, self.kspace_crop)
            if prediction is not None and prediction.ndim > 2:
                prediction = self.cropping(prediction, self.kspace_crop)
            if not self.kspace_crop:
                kspace = fft.ifft2(
                    kspace,
                    centered=self.fft_centered,
                    normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                )
            kspace = self.cropping(kspace)
            if not self.kspace_crop:
                kspace = fft.fft2(
                    kspace,
                    centered=self.fft_centered,
                    normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                )

        if not utils.is_none(mask):
            if isinstance(mask, list):
                if len(mask) == 0:
                    mask = None
            elif mask.dim() == 0:
                mask = None

        masked_kspace, mask, acc = self.masking(kspace, mask, (acq_start, acq_end), seed)  # type: ignore

        if self.cropping is not None and not self.crop_before_masking:
            if isinstance(masked_kspace, list):
                cropped_masked_kspace = []
                cropped_mask = []
                for masked_kspace_, mask_ in zip(masked_kspace, mask):
                    if not self.kspace_crop:
                        masked_kspace_ = fft.ifft2(
                            masked_kspace_,
                            centered=self.fft_centered,
                            normalization=self.fft_normalization,
                            spatial_dims=self.spatial_dims,
                        )
                    masked_kspace_ = self.cropping(masked_kspace_)
                    if not self.kspace_crop:
                        masked_kspace_ = fft.fft2(
                            masked_kspace_,
                            centered=self.fft_centered,
                            normalization=self.fft_normalization,
                            spatial_dims=self.spatial_dims,
                        )
                    cropped_masked_kspace.append(masked_kspace_)
                    cropped_mask.append(self.cropping(mask_.squeeze(-1)).unsqueeze(-1))
                masked_kspace = cropped_masked_kspace
                mask = cropped_mask
            else:
                if not self.kspace_crop:
                    masked_kspace = fft.ifft2(
                        masked_kspace,  # type: ignore
                        centered=self.fft_centered,
                        normalization=self.fft_normalization,
                        spatial_dims=self.spatial_dims,
                    )
                masked_kspace = self.cropping(masked_kspace)
                if not self.kspace_crop:
                    masked_kspace = fft.fft2(
                        masked_kspace,
                        centered=self.fft_centered,
                        normalization=self.fft_normalization,
                        spatial_dims=self.spatial_dims,
                    )
                mask = self.cropping(mask.squeeze(-1)).unsqueeze(-1)  # type: ignore

        if self.normalize_inputs:
            kspace = self.normalization(kspace, apply_backward_transform=True)
            if isinstance(masked_kspace, list):
                masked_kspace = [self.normalization(x, apply_backward_transform=True) for x in masked_kspace]
            else:
                masked_kspace = self.normalization(masked_kspace, apply_backward_transform=True)
            if sensitivity_map.size != 0:
                sensitivity_map = self.normalization(sensitivity_map, apply_backward_transform=False)
            target = self.normalization(target, apply_backward_transform=False)

        if prediction.dim() < 2:
            if isinstance(masked_kspace, list):
                prediction = [
                    utils.coil_combination_method(
                        fft.ifft2(y, self.fft_centered, self.fft_normalization, self.spatial_dims),
                        sensitivity_map,
                        self.coil_combination_method,
                        self.coil_dim,
                    )
                    for y in masked_kspace
                ]
            else:
                prediction = utils.coil_combination_method(
                    fft.ifft2(masked_kspace, self.fft_centered, self.fft_normalization, self.spatial_dims),
                    sensitivity_map,
                    self.coil_combination_method,
                    self.coil_dim,
                )

        return kspace, masked_kspace, sensitivity_map, mask, prediction, target, fname, slice_idx, acc
