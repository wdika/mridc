# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from omegaconf import DictConfig

import mridc.collections.reconstruction.nn as reconstruction_nn
from mridc.collections.common.data import subsample
from mridc.collections.common.parts import utils
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

__all__ = ["RSMRIDataTransforms"]


class RSMRIDataTransforms:
    """
    Data transforms MRI segmentation.

    Parameters
    ----------
    complex_data : bool, optional
        Whether to use complex data. If ``False`` the data are assumed to be magnitude only. Default is ``True``.
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
    RSMRIDataTransforms
        Data transformed for accelerated-MRI reconstruction and MRI segmentation.
    """

    def __init__(  # noqa: W0221
        self,
        complex_data: bool = True,
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
        self.complex_data = complex_data

        self.normalize_inputs = normalize_inputs

        self.fft_centered = fft_centered
        self.fft_normalization = fft_normalization
        self.spatial_dims = spatial_dims if spatial_dims is not None else [-2, -1]

        if not self.complex_data:
            if not utils.is_none(coil_combination_method):
                raise ValueError("Coil combination method for non-complex data should be None.")
            if not utils.is_none(mask_func):
                raise ValueError("Mask function for non-complex data should be None.")
            if kspace_crop:
                raise ValueError("K-space crop for non-complex data should be None.")
            if not utils.is_none(kspace_zero_filling_size):
                raise ValueError("K-space zero filling size for non-complex data should be None.")
            if not utils.is_none(coil_dim):
                raise ValueError("Coil dimension for non-complex data should be None.")
            if apply_prewhitening:
                raise ValueError("Prewhitening for non-complex data cannot be applied.")
            if apply_gcc:
                raise ValueError("GCC for non-complex data cannot be applied.")
        else:
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

            self.kspace_crop = kspace_crop
            self.crop_before_masking = crop_before_masking

            self.coil_combination_method = coil_combination_method
            self.coil_dim = coil_dim - 1

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
        imspace: np.ndarray,
        sensitivity_map: np.ndarray,
        mask: np.ndarray,
        initial_prediction_reconstruction: np.ndarray,
        segmentation_labels: np.ndarray,
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
        Union[Optional[torch.Tensor], Any],
        str,
        int,
        Union[List, Any],
    ]:
        """
        Apply the data transform.

        Parameters
        ----------
        kspace: The kspace.
        imspace: The image space.
        sensitivity_map: The sensitivity map.
        mask: List, sampling mask if exists and brain mask and head mask.
        initial_prediction_reconstruction: The initial reconstruction prediction.
        segmentation_labels: The segmentation labels.
        attrs: The attributes.
        fname: The file name.
        slice_idx: The slice number.

        Returns
        -------
        The transformed data.
        """
        initial_prediction_reconstruction = (
            utils.to_tensor(initial_prediction_reconstruction)
            if initial_prediction_reconstruction is not None and initial_prediction_reconstruction.size != 0
            else torch.tensor([])
        )

        if not self.complex_data:
            imspace = torch.from_numpy(imspace)
            initial_prediction_reconstruction = torch.abs(imspace)
            target_reconstruction = imspace
            kspace = torch.empty([])
            sensitivity_map = torch.empty([])
            masked_kspace = torch.empty([])
            mask = torch.empty([])
            acc = torch.empty([])
        else:
            kspace, masked_kspace, mask, acc = self.__process_kspace__(kspace, mask, attrs, fname)
            sensitivity_map = self.__process_coil_sensitivities_map__(sensitivity_map, kspace)
            target_reconstruction = self.__initialize_prediction__(torch.empty([]), kspace, sensitivity_map)
            initial_prediction_reconstruction = self.__initialize_prediction__(
                initial_prediction_reconstruction, masked_kspace, sensitivity_map
            )
            if isinstance(initial_prediction_reconstruction, list):
                initial_prediction_reconstruction = [
                    torch.view_as_real(x) for x in initial_prediction_reconstruction if x.shape[-1] != 2
                ]
            elif initial_prediction_reconstruction.shape[-1] != 2:
                initial_prediction_reconstruction = torch.view_as_real(initial_prediction_reconstruction)

        if not utils.is_none(segmentation_labels) and segmentation_labels.ndim > 1:
            segmentation_labels = self.cropping(torch.from_numpy(segmentation_labels))  # type: ignore
        else:
            segmentation_labels = torch.empty([])
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

    def __repr__(self) -> str:
        return (
            f"Preprocessing transforms initialized for {self.__class__.__name__}: "
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

    def __process_kspace__(self, kspace: np.ndarray, mask: Union[np.ndarray, None], attrs: Dict, fname: str) -> Tuple[
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
        self, prediction: Union[torch.Tensor, np.ndarray, None], kspace: torch.Tensor, sensitivity_map: torch.Tensor
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
