# coding=utf-8
__author__ = "Dimitrios Karkalousos, Chaoping Zhang"

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from skimage.restoration import unwrap_phase
from torch import Tensor
from torch.nn import functional as F

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

__all__ = ["qMRIDataTransforms"]


class qMRIDataTransforms:
    """
    Data transforms for quantitative MRI.

    Parameters
    ----------
    TEs : Optional[List[float]]
        Echo times.
    precompute_quantitative_maps : bool, optional
        Precompute quantitative maps. Default is ``True``.
    qmaps_scaling_factor : float, optional
        Quantitative maps scaling factor. Default is ``1e-3``.
    shift_B0_input : bool, optional
        Whether to shift the B0 input. Default is ``False``.
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
    qMRIDataTransforms
        Data transformed for quantitative MRI.
    """

    def __init__(
        self,
        TEs: Optional[List[float]],
        precompute_quantitative_maps: bool = True,
        qmaps_scaling_factor: float = 1.0,
        shift_B0_input: bool = False,
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

        if TEs is None:
            raise ValueError("Please specify echo times (TEs).")
        self.TEs = TEs
        self.precompute_quantitative_maps = precompute_quantitative_maps

        self.coil_combination_method = coil_combination_method
        self.kspace_crop = kspace_crop
        self.crop_before_masking = crop_before_masking
        self.normalize_inputs = normalize_inputs

        self.fft_centered = fft_centered
        self.fft_normalization = fft_normalization
        self.spatial_dims = spatial_dims if spatial_dims is not None else [-2, -1]
        self.coil_dim = coil_dim - 1

        self.qmaps_scaling_factor = qmaps_scaling_factor
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
        qmaps: np.ndarray,
        mask: np.ndarray,
        prediction: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_idx: int,
    ) -> Tuple[
        Union[Union[List, torch.Tensor], torch.Tensor],
        Union[Optional[torch.Tensor], Any],
        Union[Union[List, torch.Tensor], torch.Tensor],
        Union[Optional[torch.Tensor], Any],
        Union[Union[List, torch.Tensor], torch.Tensor],
        Union[Optional[torch.Tensor], Any],
        Union[Union[List, torch.Tensor], torch.Tensor],
        Union[Optional[torch.Tensor], Any],
        torch.Tensor,
        torch.Tensor,
        Union[Union[List, torch.Tensor], torch.Tensor],
        Union[Optional[torch.Tensor], Any],
        Union[List, Any],
        torch.Tensor,
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
        qmaps: The quantitative maps.
        mask: List, sampling mask if exists and brain mask and head mask.
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

        mask_head = mask[2]
        mask_brain = mask[1]
        mask = mask[0]

        if mask_brain.ndim != 0:
            mask_brain = torch.from_numpy(mask_brain)

        if mask_head.ndim != 0:
            mask_head = torch.from_numpy(mask_head)

        if mask is not None:
            if isinstance(mask, list):
                mask = [torch.from_numpy(m) for m in mask]
            elif mask.ndim != 0:
                mask = torch.from_numpy(mask)

        if self.prewhitening is not None:
            # todo: dynamic echo dim
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

            if mask_brain.dim() != 0:
                mask_brain = self.cropping(mask_brain)

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

        mask_head = torch.ones_like(mask_brain)

        if self.precompute_quantitative_maps:
            R2star_maps_init = []
            S0_maps_init = []
            B0_maps_init = []
            phi_maps_init = []
            predictions = []
            for y in masked_kspace:
                prediction = utils.sense(
                    fft.ifft2(
                        y,
                        centered=self.fft_centered,
                        normalization=self.fft_normalization,
                        spatial_dims=self.spatial_dims,
                    ),
                    sensitivity_map.unsqueeze(0),
                    dim=self.coil_dim,
                )

                predictions.append(prediction)
                R2star_map_init, S0_map_init, B0_map_init, phi_map_init = R2star_B0_S0_phi_mapping(
                    prediction,
                    self.TEs,  # type: ignore
                    mask_brain,
                    mask_head,
                    fully_sampled=True,
                    scaling_factor=self.qmaps_scaling_factor,
                    shift=self.shift_B0_input,
                    fft_centered=self.fft_centered,
                    fft_normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                )

                R2star_maps_init.append(R2star_map_init)
                S0_maps_init.append(S0_map_init)
                B0_maps_init.append(B0_map_init)
                phi_maps_init.append(phi_map_init)

            R2star_map_init = R2star_maps_init
            S0_map_init = S0_maps_init
            B0_map_init = B0_maps_init
            phi_map_init = phi_maps_init

            mask_brain_tmp = torch.ones_like(torch.abs(mask_brain))
            mask_brain_tmp = mask_brain_tmp.unsqueeze(0) if mask_brain.dim() == 2 else mask_brain_tmp
            imspace = utils.sense(
                fft.ifft2(
                    kspace,
                    centered=self.fft_centered,
                    normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                )
                * mask_brain_tmp.unsqueeze(self.coil_dim - 1).unsqueeze(-1),
                sensitivity_map.unsqueeze(0),
                dim=self.coil_dim,
            )

            if qmaps[0][-1].ndim != 0:
                B0_map, S0_map, R2star_map, phi_map = qmaps
                B0_map_target = B0_map[-1]
                S0_map_target = S0_map[-1]
                R2star_map_target = R2star_map[-1]
                phi_map_target = phi_map[-1]
            else:
                R2star_map_target, S0_map_target, B0_map_target, phi_map_target = R2star_B0_S0_phi_mapping(
                    imspace,
                    self.TEs,  # type: ignore
                    mask_brain,
                    mask_head,
                    fully_sampled=True,
                    scaling_factor=self.qmaps_scaling_factor,
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
                raise ValueError(
                    "No quantitative maps were found, while the precompute_quantitative_maps flag is set to False."
                    "Please either set the precompute_quantitative_maps flag to True or precompute and store your "
                    "quantitative maps in your input data."
                )

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
            prediction,
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
        kernel_size: Union[List[int], int],
        sigma: float,
        dim: int = 2,
        shift: bool = False,
        fft_centered: bool = False,
        fft_normalization: str = "backward",
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
        kernel = torch.exp(
            -0.5
            * (
                (torch.arange(kernel_size[0])[:, None] - (kernel_size[0] - 1) / 2) ** 2 / sigma[0] ** 2  # type: ignore
                + (torch.arange(kernel_size[1])[None, :] - (kernel_size[1] - 1) / 2) ** 2 / sigma[1] ** 2  # type: ignore
            )
        )
        kernel /= kernel.sum()

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
            raise RuntimeError(f"Only 1, 2 and 3 dimensions are supported. Received {dim}.")

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
            input = fft.ifft2(
                torch.fft.fftshift(
                    fft.fft2(
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
            x = fft.ifft2(
                torch.fft.fftshift(
                    fft.fft2(
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


class LeastSquaresFitting:
    def __init__(self, device):
        super(LeastSquaresFitting, self).__init__()
        self.device = device

    @staticmethod
    def lsqrt(A: torch.Tensor, Y: torch.Tensor, reg_factor: float = 0.0) -> torch.Tensor:
        """
        Differentiable least square solution.

        Parameters
        ----------
        A : torch.Tensor
            Input matrix.
        Y : torch.Tensor
            Echo times matrix.
        reg_factor : float
            Regularization parameter.

        Returns
        -------
        torch.Tensor
            Least square solution.
        """
        q, r = torch.qr(A)
        return torch.inverse(r) @ q.permute(0, 2, 1) @ Y + reg_factor

    @staticmethod
    def lsqrt_pinv(A: torch.Tensor, Y: torch.Tensor, reg_factor: float = 0.0) -> torch.Tensor:
        """
        Differentiable inverse least square solution.

        Parameters
        ----------
        A : torch.Tensor
            Input matrix.
        Y : torch.Tensor
            Echo times matrix.
        reg_factor : float
            Regularization parameter.

        Returns
        -------
        torch.Tensor
            Inverse least square solution.
        """
        if Y.dim() == 2:
            return torch.matmul(torch.inverse(Y), A)
        return torch.matmul(
            torch.matmul(torch.inverse(torch.matmul(Y.permute(0, 2, 1), Y) + reg_factor), Y.permute(0, 2, 1)), A
        )


def R2star_B0_S0_phi_mapping(
    prediction: torch.Tensor,
    TEs: Union[Optional[List[float]], float],
    brain_mask: torch.Tensor,
    head_mask: torch.Tensor,
    fully_sampled: bool = True,
    scaling_factor: float = 1e-3,
    shift: bool = False,
    fft_centered: bool = False,
    fft_normalization: str = "backward",
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
    scaling_factor : float
        The scaling factor to apply to the prediction.
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
    R2star_map = R2star_mapping(prediction, TEs, scaling_factor=scaling_factor)
    B0_map = -B0_phi_mapping(
        prediction,
        TEs,
        brain_mask,
        head_mask,
        fully_sampled,
        scaling_factor=scaling_factor,
        shift=shift,
        fft_centered=fft_centered,
        fft_normalization=fft_normalization,
        spatial_dims=spatial_dims,
    )[0]
    S0_map, phi_map = S0_mapping(
        prediction,
        TEs,
        R2star_map,
        B0_map,
        scaling_factor=scaling_factor,
        shift=shift,
        fft_centered=fft_centered,
        fft_normalization=fft_normalization,
        spatial_dims=spatial_dims,
    )
    return R2star_map, S0_map, B0_map, phi_map


def R2star_mapping(
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
    prediction_flatten = prediction.flatten(start_dim=1, end_dim=-1).cpu()
    TEs = torch.tensor(TEs).to(prediction_flatten) * scaling_factor
    TEs = torch.stack([TEs, torch.ones_like(TEs)], dim=1).T

    sqrt_prediction = torch.sqrt(prediction_flatten)
    b = torch.matmul(TEs, torch.log(prediction_flatten) * sqrt_prediction)
    A = [torch.matmul(TEs, TEs.T * sqrt_prediction[:, i, None]) for i in range(prediction_flatten.shape[1])]  # type: ignore

    R2star_map = torch.empty([prediction_flatten.shape[1]])
    for i in range(prediction_flatten.shape[1]):
        R2star_map[i] = torch.linalg.solve(A[i], b[:, i])[0]
    R2star_map = -R2star_map.detach().reshape(prediction.shape[1:4]).to(prediction)

    return R2star_map


def B0_phi_mapping(
    prediction: torch.Tensor,
    TEs: Union[Optional[List[float]], float],
    brain_mask: torch.Tensor,
    head_mask: torch.Tensor,
    fully_sampled: bool = True,
    scaling_factor: float = 1e-3,
    shift: bool = False,
    fft_centered: bool = False,
    fft_normalization: str = "backward",
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
    lsq = LeastSquaresFitting(device=prediction.device)

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
        prediction = fft.ifft2(
            torch.fft.fftshift(fft.fft2(prediction, fft_centered, fft_normalization, spatial_dims), dim=(1, 2)),
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
    B0_map_tmp = lsq.lsqrt_pinv(
        phase_diff_set.unsqueeze(2).permute(1, 0, 2), TE_diff.unsqueeze(1) * scaling_factor  # type: ignore
    )
    B0_map = B0_map_tmp.reshape(shape[-3], shape[-2])
    B0_map = B0_map * torch.abs(head_mask)

    # obtain phi map
    phi_map = (phase_unwrapped[0] - scaling_factor * TEs[0] * B0_map).squeeze(0)  # type: ignore

    return B0_map.to(prediction), phi_map.to(prediction)


def S0_mapping(
    prediction: torch.Tensor,
    TEs: Union[Optional[List[float]], float],
    R2star_map: torch.Tensor,
    B0_map: torch.Tensor,
    scaling_factor: float = 1e-3,
    shift: bool = False,
    fft_centered: bool = False,
    fft_normalization: str = "backward",
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
    lsq = LeastSquaresFitting(device=prediction.device)

    prediction = torch.view_as_complex(prediction)
    prediction_flatten = prediction.reshape(prediction.shape[0], -1)

    TEs = torch.tensor(TEs).to(prediction)

    R2star_B0_complex_map = R2star_map.to(prediction) + 1j * B0_map.to(prediction)
    R2star_B0_complex_map_flatten = R2star_B0_complex_map.flatten()

    TEs_r2 = TEs[0:4].unsqueeze(1) * -R2star_B0_complex_map_flatten  # type: ignore

    S0_map = lsq.lsqrt_pinv(
        prediction_flatten.permute(1, 0).unsqueeze(2),
        torch.exp(scaling_factor * 1e-3 * TEs_r2.permute(1, 0).unsqueeze(2)),
    )

    S0_map = torch.view_as_real(S0_map.reshape(prediction.shape[1:]))

    if shift:
        S0_map = fft.ifft2(
            torch.fft.fftshift(fft.fft2(S0_map, fft_centered, fft_normalization, spatial_dims), dim=(0, 1)),
            fft_centered,
            fft_normalization,
            spatial_dims,
        )

    S0_map_real, S0_map_imag = torch.chunk(S0_map, 2, dim=-1)

    return S0_map_real.squeeze(-1), S0_map_imag.squeeze(-1)
