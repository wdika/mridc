# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from omegaconf import DictConfig
from skimage.restoration import unwrap_phase
from torch.nn import functional as F

import mridc.collections.reconstruction.nn as reconstruction_nn
from mridc.collections.common.data import subsample
from mridc.collections.common.parts import fft, utils
from mridc.collections.common.parts.transforms import (
    SSDU,
    Composer,
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
    find_patch_size : bool, optional
        Find optimal patch size (automatically) to calculate psi. If False, patch_size must be defined.
        Default is ``True``.
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
    ssdu : bool, optional
        Whether to apply Self-Supervised Data Undersampling (SSDU) masks. Default is ``False``.
    ssdu_mask_type: str, optional
        Mask type. It can be one of the following:
        - "Gaussian": Gaussian sampling.
        - "Uniform": Uniform sampling.
        Default is "Gaussian".
    ssdu_rho: float, optional
        Split ratio for training and loss masks. Default is ``0.4``.
    ssdu_acs_block_size: tuple, optional
        Keeps a small acs region fully-sampled for training masks, if there is no acs region. The small acs block
        should be set to zero. Default is ``(4, 4)``.
    ssdu_gaussian_std_scaling_factor: float, optional
        Scaling factor for standard deviation of the Gaussian noise. If Uniform is select this factor is ignored.
        Default is ``4.0``.
    ssdu_outer_kspace_fraction: float, optional
        Fraction of the outer k-space to be kept/unmasked. Default is ``0.05``.
    ssdu_export_and_reuse_masks: bool, optional
        Whether to export and reuse the masks. Default is ``False``.
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
    kspace_normalization : bool, optional
        Whether to normalize the k-space. Default is ``False``.
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

    def __init__(  # noqa: W0221
        self,
        TEs: Optional[List[float]],
        precompute_quantitative_maps: bool = True,
        qmaps_scaling_factor: float = 1.0,
        shift_B0_input: bool = False,
        apply_prewhitening: bool = False,
        find_patch_size: bool = True,
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
        ssdu: bool = False,
        ssdu_mask_type: str = "Gaussian",
        ssdu_rho: float = 0.4,
        ssdu_acs_block_size: Sequence[int] = (4, 4),
        ssdu_gaussian_std_scaling_factor: float = 4.0,
        ssdu_outer_kspace_fraction: float = 0.05,
        ssdu_export_and_reuse_masks: bool = False,
        crop_size: Optional[Tuple[int, int]] = None,
        kspace_crop: bool = False,
        crop_before_masking: bool = True,
        kspace_zero_filling_size: Optional[Tuple] = None,
        normalize_inputs: bool = True,
        normalization_type: str = "max",
        kspace_normalization: bool = False,
        fft_centered: bool = False,
        fft_normalization: str = "backward",
        spatial_dims: Sequence[int] = None,
        coil_dim: int = 0,
        consecutive_slices: int = 1,  # noqa: W0613
        use_seed: bool = True,
    ):
        super().__init__()

        if TEs is None:
            raise ValueError("Please specify echo times (TEs).")
        self.TEs = TEs
        self.precompute_quantitative_maps = precompute_quantitative_maps
        self.qmaps_scaling_factor = qmaps_scaling_factor
        self.shift_B0_input = shift_B0_input

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
                find_patch_size=find_patch_size,
                patch_size=[
                    prewhitening_patch_start,
                    prewhitening_patch_length + prewhitening_patch_start,
                    prewhitening_patch_start,
                    prewhitening_patch_length + prewhitening_patch_start,
                ],
                scale_factor=prewhitening_scale_factor,
                fft_centered=self.fft_centered,
                fft_normalization=self.fft_normalization,
                spatial_dims=self.spatial_dims,
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
                fft_centered=self.fft_centered,
                fft_normalization=self.fft_normalization,
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

        self.ssdu = ssdu
        self.ssdu_masking = (
            SSDU(
                mask_type=ssdu_mask_type,
                rho=ssdu_rho,
                acs_block_size=ssdu_acs_block_size,
                gaussian_std_scaling_factor=ssdu_gaussian_std_scaling_factor,
                outer_kspace_fraction=ssdu_outer_kspace_fraction,
                export_and_reuse_masks=ssdu_export_and_reuse_masks,
            )
            if self.ssdu
            else None
        )

        self.cropping = (
            Cropper(
                cropping_size=crop_size,  # type: ignore
                fft_centered=self.fft_centered,  # type: ignore
                fft_normalization=self.fft_normalization,  # type: ignore
                spatial_dims=self.spatial_dims,  # type: ignore
            )
            if not utils.is_none(crop_size)
            else None
        )

        self.normalization = Normalizer(
            normalization_type=normalization_type,
            kspace_normalization=kspace_normalization,
            fft_centered=self.fft_centered,
            fft_normalization=self.fft_normalization,
            spatial_dims=self.spatial_dims,  # type: ignore
        )

        self.init_reconstructor = reconstruction_nn.zf.ZF(  # type: ignore
            cfg=DictConfig(
                {
                    "fft_centered": self.fft_centered,
                    "fft_normalization": self.fft_normalization,
                    "spatial_dims": self.spatial_dims,
                    "coil_dim": self.coil_dim,
                    "coil_combination_method": self.coil_combination_method.upper(),
                }
            )
        )
        self.prewhitening = Composer([self.prewhitening])  # type: ignore
        self.coils_shape_transforms = Composer(
            [
                self.gcc,  # type: ignore
                self.kspace_zero_filling,  # type: ignore
            ]
        )
        self.crop_normalize = Composer(
            [
                self.cropping,  # type: ignore
                self.normalization,  # type: ignore
            ]
        )
        self.cropping = Composer([self.cropping])  # type: ignore
        self.normalization = Composer([self.normalization])  # type: ignore

        self.use_seed = use_seed

    def __call__(  # noqa: W0221
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
        List[Any],
        Any,
        List[Any],
        Any,
        List[Any],
        Any,
        List[Any],
        Any,
        torch.Tensor,
        torch.Tensor,
        Union[List[torch.Tensor], torch.Tensor],
        torch.Tensor,
        Union[List[torch.Tensor], torch.Tensor],
        torch.Tensor | Any,
        Union[List[torch.Tensor], torch.Tensor],
        Union[List[torch.Tensor], torch.Tensor],
        str,
        int,
        List[Union[float, torch.Tensor, Any]],
    ]:
        """
        Apply the data transform.

        Parameters
        ----------
        kspace: The kspace.
        sensitivity_map: The sensitivity map.
        qmaps: The quantitative maps.
        mask: List of masks, with the undersampling, the brain, and the head mask.
        prediction: The initial estimation.
        target: The target.
        attrs: The attributes.
        fname: The file name.
        slice_idx: The slice number.

        Returns
        -------
        The transformed data.
        """
        mask, mask_brain, mask_head = mask

        kspace, masked_kspace, mask, acc = self.__process_kspace__(kspace, mask, attrs, fname)
        sensitivity_map = self.__process_coil_sensitivities_map__(sensitivity_map, kspace)
        target = self.__initialize_prediction__(torch.empty([]), kspace, sensitivity_map)
        if target.shape[-1] == 2:
            target = torch.view_as_complex(target)
        prediction = self.__initialize_prediction__(prediction, masked_kspace, sensitivity_map)

        if mask_brain.ndim != 0:
            mask_brain = self.crop_normalize(torch.from_numpy(mask_brain))
            mask_head = torch.ones_like(mask_brain)

        if self.precompute_quantitative_maps:
            (
                R2star_map_init,
                R2star_map_target,
                S0_map_init,
                S0_map_target,
                B0_map_init,
                B0_map_target,
                phi_map_init,
                phi_map_target,
                prediction,
            ) = self.__compute_quantitative_maps__(kspace, masked_kspace, sensitivity_map, mask_brain, mask_head)
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

    def __compute_quantitative_maps__(  # noqa: W0221
        self,
        kspace: torch.Tensor,
        masked_kspace: torch.Tensor,
        sensitivity_map: torch.Tensor,
        mask_brain: torch.Tensor,
        mask_head: torch.Tensor,
    ) -> Tuple[List[Any], Any, List[Any], Any, List[Any], Any, List[Any], Any, List[torch.Tensor]]:
        """
        Compute quantitative maps from the masked kspace data.

        Parameters
        ----------
        kspace: torch.Tensor
            Kspace data.
        masked_kspace: torch.Tensor
            Masked kspace data.
        sensitivity_map: torch.Tensor
            Sensitivity maps.
        mask_brain: torch.Tensor
            Brain mask.
        mask_head: torch.Tensor
            Head mask.

        Returns
        -------
        R2star_maps_init: List[torch.Tensor]
            List of R2star maps.
        S0_maps_init: List[torch.Tensor]
            List of S0 maps.
        B0_maps_init: List[torch.Tensor]
            List of B0 maps.
        phi_maps_init: List[torch.Tensor]
            List of phi maps.
        """
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

        return (
            R2star_map_init,
            R2star_map_target,
            S0_map_init,
            S0_map_target,
            B0_map_init,
            B0_map_target,
            phi_map_init,
            phi_map_target,
            predictions,
        )

    def __repr__(self) -> str:
        return (
            f"Preprocessing transforms initialized for {self.__class__.__name__}: "
            f"precompute_quantitative_maps = {self.precompute_quantitative_maps}, "
            f"prewhitening = {self.prewhitening}, "
            f"masking = {self.masking}, "
            f"SSDU masking = {self.ssdu_masking}, "
            f"kspace zero-filling = {self.kspace_zero_filling}, "
            f"cropping = {self.cropping}, "
            f"normalization = {self.normalization}, "
            f"initial reconstructor = {self.init_reconstructor}, "
        )

    def __str__(self) -> str:
        return self.__repr__()

    def __process_kspace__(
        self, kspace: np.ndarray, mask: Union[np.ndarray, None], attrs: Dict, fname: str
    ) -> Tuple[
        torch.Tensor,
        Union[List[torch.Tensor], torch.Tensor],
        Union[List[torch.Tensor], torch.Tensor],
        Union[List[Union[float, torch.Tensor, Any]]],
    ]:
        """
        Apply the preprocessing transforms to the kspace.

        Parameters
        ----------
        kspace : torch.Tensor
            The kspace.
        mask : torch.Tensor
            The mask, if None, the mask is generated.
        attrs : Dict
            The attributes, if stored in the file.
        fname : str
            The file name.

        Returns
        -------
        The preprocessed kspace.
        """
        kspace = utils.to_tensor(kspace)
        kspace = utils.add_coil_dim_if_singlecoil(kspace, dim=self.coil_dim)

        kspace = self.coils_shape_transforms(kspace, apply_backward_transform=True)
        kspace = self.prewhitening(kspace)  # type: ignore

        if self.crop_before_masking:
            kspace = self.cropping(kspace, apply_backward_transform=not self.kspace_crop)  # type: ignore

        masked_kspace, mask, acc = self.masking(
            kspace,
            mask,
            (
                attrs["padding_left"] if "padding_left" in attrs else 0,
                attrs["padding_right"] if "padding_right" in attrs else 0,
            ),
            tuple(map(ord, fname)) if self.use_seed else None,  # type: ignore
        )

        if not self.crop_before_masking:
            masked_kspace = self.cropping(masked_kspace, apply_backward_transform=not self.kspace_crop)  # type: ignore
            if isinstance(mask, list):
                mask = [self.cropping(x.squeeze(-1)).unsqueeze(-1) for x in mask]  # type: ignore
            kspace = self.cropping(kspace, apply_backward_transform=not self.kspace_crop)  # type: ignore

        kspace = self.normalization(kspace, apply_backward_transform=True)
        masked_kspace = self.normalization(masked_kspace, apply_backward_transform=True)

        if self.ssdu:
            if isinstance(mask, list):
                kspaces = []
                masked_kspaces = []
                masks = []
                for i in range(len(mask)):  # noqa: C0200
                    train_mask, loss_mask = self.ssdu_masking(kspace, mask[i], fname)  # type: ignore  # noqa: E1102
                    kspaces.append(kspace * loss_mask)
                    masked_kspaces.append(masked_kspace[i] * train_mask)  # type: ignore
                    masks.append([train_mask, loss_mask])
                kspace = kspaces
                masked_kspace = masked_kspaces
                mask = masks
            else:
                train_mask, loss_mask = self.ssdu_masking(kspace, mask, fname)  # type: ignore  # noqa: E1102
                kspace = kspace * loss_mask
                masked_kspace = masked_kspace * train_mask
                mask = [train_mask, loss_mask]

        return kspace, masked_kspace, mask, acc

    def __process_coil_sensitivities_map__(self, sensitivity_map: np.ndarray, kspace: torch.Tensor) -> torch.Tensor:
        """
        Preprocesses the coil sensitivities map.

        Parameters
        ----------
        sensitivity_map : np.ndarray
            The coil sensitivities map.
        kspace : torch.Tensor
            The kspace.

        Returns
        -------
        torch.Tensor
            The preprocessed coil sensitivities map.
        """
        # This condition is necessary in case of auto estimation of sense maps.
        if sensitivity_map is not None and sensitivity_map.size != 0:
            sensitivity_map = utils.to_tensor(sensitivity_map)
        else:
            # If no sensitivity map is provided, either the data is singlecoil or the sense net is used.
            # Initialize the sensitivity map to 1 to assure for the singlecoil case.
            sensitivity_map = torch.ones_like(kspace)
        sensitivity_map = self.coils_shape_transforms(sensitivity_map, apply_forward_transform=True)
        sensitivity_map = self.crop_normalize(sensitivity_map, apply_forward_transform=self.kspace_crop)
        return sensitivity_map

    def __initialize_prediction__(
        self, prediction: Union[np.ndarray, torch.Tensor, None], kspace: torch.Tensor, sensitivity_map: torch.Tensor
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Predicts a coil-combined image.

        Parameters
        ----------
        prediction : np.ndarray
            The initial estimation, if None, the prediction is initialized.
        kspace : torch.Tensor
            The kspace.
        sensitivity_map : torch.Tensor
            The sensitivity map.

        Returns
        -------
        Union[List[torch.Tensor], torch.Tensor]
            The initialized prediction, either a list of coil-combined images or a single coil-combined image.
        """
        if utils.is_none(prediction) or prediction.ndim < 2 or isinstance(kspace, list):  # type: ignore
            if isinstance(kspace, list):
                prediction = [
                    self.crop_normalize(
                        self.init_reconstructor(y, sensitivity_map, torch.empty([]), torch.empty([]), torch.empty([]))
                    )
                    for y in kspace
                ]
            else:
                prediction = self.crop_normalize(
                    self.init_reconstructor(kspace, sensitivity_map, torch.empty([]), torch.empty([]), torch.empty([]))
                )
        else:
            if isinstance(prediction, np.ndarray):
                prediction = utils.to_tensor(prediction)
            prediction = self.crop_normalize(prediction, apply_forward_transform=self.kspace_crop)
            if prediction.shape[-1] != 2 and prediction.type == "torch.ComplexTensor":  # type: ignore
                prediction = torch.view_as_real(prediction)
        return prediction


class GaussianSmoothing(torch.nn.Module):
    """
    Apply gaussian smoothing on a 1d, 2d or 3d tensor. Filtering is performed separately for each channel in the input
    using a depthwise convolution.
    """

    def __init__(  # noqa: W0221
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
        super().__init__()

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
                + (torch.arange(kernel_size[1])[None, :] - (kernel_size[1] - 1) / 2) ** 2
                / sigma[1] ** 2  # type: ignore
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

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Apply gaussian filter to input.

        Parameters
        ----------
        data : torch.Tensor
            Input to apply gaussian filter on.

        Returns
        -------
        torch.Tensor
            Filtered output.
        """
        if self.shift:
            data = data.permute(0, 2, 3, 1)
            data = fft.ifft2(
                torch.fft.fftshift(
                    fft.fft2(
                        torch.view_as_real(data[..., 0] + 1j * data[..., 1]),
                        self.fft_centered,
                        self.fft_normalization,
                        self.spatial_dims,
                    ),
                ),
                self.fft_centered,
                self.fft_normalization,
                self.spatial_dims,
            ).permute(0, 3, 1, 2)

        x = self.conv(data, weight=self.weight.to(data), groups=self.groups).to(data).detach()

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
        super().__init__()
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


def R2star_B0_S0_phi_mapping(  # noqa: W0221
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
    A = [
        torch.matmul(TEs, TEs.T * sqrt_prediction[:, i, None])  # type: ignore
        for i in range(prediction_flatten.shape[1])
    ]

    R2star_map = torch.empty([prediction_flatten.shape[1]])
    for i in range(prediction_flatten.shape[1]):
        R2star_map[i] = torch.linalg.solve(A[i], b[:, i])[0]
    R2star_map = -R2star_map.detach().reshape(prediction.shape[1:4]).to(prediction)

    return R2star_map


def B0_phi_mapping(  # noqa: W0221
    prediction: torch.Tensor,
    TEs: Union[Optional[List[float]], float],
    brain_mask: torch.Tensor,
    head_mask: torch.Tensor,
    fully_sampled: bool = True,  # noqa: W0613
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


def S0_mapping(  # noqa: W0221
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
