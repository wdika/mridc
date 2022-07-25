# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Parts of the code have been taken from https://github.com/facebookresearch/fastMRI

from math import sqrt
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from mridc.collections.common.parts.fft import fft2, ifft2
from mridc.collections.common.parts.utils import is_none, reshape_fortran, rss, sense, to_tensor
from mridc.collections.reconstruction.data.subsample import MaskFunc
from mridc.collections.reconstruction.parts.utils import apply_mask, center_crop, complex_center_crop

__all__ = ["MRIDataTransforms"]


class MRIDataTransforms:
    """MRI preprocessing data transforms."""

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
        use_seed: bool = True,
    ):
        """
        Initialize the data transform.

        Parameters
        ----------
        coil_combination_method : The coil combination method to use. Default: 'SENSE'
        dimensionality : The dimensionality of the data. Default: 2
        mask_func: The function that masks the kspace.
        shift_mask: Whether to shift the mask.
        mask_center_scale: The scale of the center of the mask.
        half_scan_percentage: The percentage of the scan to be used.
        remask: Whether to only generate 1 mask per set of consecutive slices of 3D data.
        crop_size: The size of the crop.
        kspace_crop: Whether to crop the kspace.
        crop_before_masking: Whether to crop before masking.
        kspace_zero_filling_size: The size of padding in kspace -> zero filling.
        normalize_inputs: Whether to normalize the inputs.
        fft_centered: Whether to center the fft.
        fft_normalization: The normalization of the fft.
        spatial_dims: The spatial dimensions of the data.
        coil_dim: The coil dimension of the data.
        use_seed: Whether to use the seed.
        """
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
        mask: np.ndarray,
        eta: np.ndarray,
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

        if self.apply_prewhitening:
            kspace = self.prewhitening(kspace)  # type: ignore

        if self.gcc is not None:
            kspace = self.gcc(kspace)
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
            padding_top = np.floor_divide(abs(int(self.kspace_zero_filling_size[0]) - kspace.shape[1]), 2)
            padding_bottom = padding_top
            padding_left = np.floor_divide(abs(int(self.kspace_zero_filling_size[1]) - kspace.shape[2]), 2)
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
                    sensitivity_map,
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
        crop_size = target.shape
        if self.crop_size is not None and self.crop_size not in ("", "None"):
            # Check for smallest size against the target shape.
            h = min(int(self.crop_size[0]), target.shape[0])
            w = min(int(self.crop_size[1]), target.shape[1])

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

        if not is_none(mask) and not is_none(self.mask_func):
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

            acc = 1
            masked_kspace = kspace * mask + 0.0  # the + 0.0 removes the sign of the zeros
        elif is_none(self.mask_func):
            masked_kspace = kspace.clone()
            acc = torch.tensor([1])

            if mask is None:
                mask = torch.ones(masked_kspace.shape[-3], masked_kspace.shape[-2]).type(torch.float32)
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

            if sensitivity_map.size != 0:
                sensitivity_map = sensitivity_map / torch.max(torch.abs(sensitivity_map))

            if eta.size != 0 and eta.ndim > 2:
                eta = eta / torch.max(torch.abs(eta))

            # if self.max_norm:
            target = target / torch.max(torch.abs(target))

        return kspace, masked_kspace, sensitivity_map, mask, eta, target, fname, slice_idx, acc


class NoisePreWhitening:
    """Apply noise pre-whitening / coil decorrelation."""

    def __init__(
        self,
        patch_size: List[int],
        scale_factor: float = 1.0,
    ):
        """
        Parameters
        ----------
        patch_size : list of ints
            Define patch size to calculate psi.
            x_start, x_end, y_start, y_end
        scale_factor : float
            Applied on the noise covariance matrix. Used to adjust for effective noise bandwidth and difference in
            sampling rate between noise calibration and actual measurement.
            scale_factor = (T_acq_dwell/T_noise_dwell)*NoiseReceiverBandwidthRatio
        """
        super().__init__()
        self.patch_size = patch_size
        self.scale_factor = scale_factor

    def __call__(self, data):
        if not self.patch_size:
            raise ValueError("Patch size must be defined for noise prewhitening.")

        if data.shape[-1] != 2:
            data = torch.view_as_real(data)

        noise = data[:, self.patch_size[0] : self.patch_size[1], self.patch_size[-2] : self.patch_size[-1]]
        noise_int = torch.reshape(noise, (noise.shape[0], int(torch.numel(noise) / noise.shape[0])))

        deformation_matrix = (1 / (float(noise_int.shape[1]) - 1)) * torch.mm(noise_int, torch.conj(noise_int).t())
        psi = torch.linalg.inv(torch.linalg.cholesky(deformation_matrix)) * sqrt(2) * sqrt(self.scale_factor)

        return torch.reshape(
            torch.mm(psi, torch.reshape(data, (data.shape[0], int(torch.numel(data) / data.shape[0])))), data.shape
        )


class GeometricDecompositionCoilCompression:
    """
    Geometric Decomposition Coil Compression Based on:
    Zhang, T., Pauly, J. M., Vasanawala, S. S., & Lustig, M. (2013). Coil compression for accelerated imaging with
    Cartesian sampling. Magnetic Resonance in Medicine, 69(2), 571–582. https://doi.org/10.1002/mrm.24267
    """

    def __init__(
        self,
        virtual_coils: int = None,
        calib_lines: int = None,
        align_data: bool = True,
        fft_centered: bool = True,
        fft_normalization: str = "ortho",
        spatial_dims: Sequence[int] = None,
    ):
        """
        Parameters
        ----------
        virtual_coils : number of final-"virtual" coils
        calib_lines : calibration lines to sample data points
        align_data : align data to the first calibration line
        fft_centered: Whether to center the fft.
        fft_normalization: "ortho" is the default normalization used by PyTorch. Can be changed to "ortho" or None.
        spatial_dims: dimensions to apply the FFT
        """
        super().__init__()
        self.virtual_coils = virtual_coils
        self.calib_lines = calib_lines
        self.align_data = align_data
        self.fft_centered = fft_centered
        self.fft_normalization = fft_normalization
        self.spatial_dims = spatial_dims

    def __call__(self, data):
        if not self.virtual_coils:
            raise ValueError("Number of virtual coils must be defined for geometric decomposition coil compression.")

        self.data = data
        if self.data.shape[-1] == 2:
            self.data = torch.view_as_complex(self.data)

        curr_num_coils = self.data.shape[0]
        if curr_num_coils < self.virtual_coils:
            raise ValueError(
                f"Tried to compress from {curr_num_coils} to {self.virtual_coils} coils, please select less coils for."
            )

        # TODO: think about if needed to handle slices here or not
        # if "slice" in self.data.names or self.data.dim() <= 3:
        #     raise ValueError(f"Currently supported only 2D data, you have given as input {self.data.dim()}D data.")

        self.data = self.data.permute(1, 2, 0)
        self.init_data: torch.Tensor = self.data
        self.fft_dim = [0, 1]

        _, self.width, self.coils = self.data.shape

        # TODO: figure out why this is happening for singlecoil data
        # For singlecoil data, use no calibration lines equal to the no of coils.
        if self.virtual_coils == 1:
            self.calib_lines = self.data.shape[-1]

        self.crop()
        self.calculate_gcc()

        if self.align_data:
            self.align_compressed_coils()
            rotated_compressed_data = self.rotate_and_compress(data_to_cc=self.aligned_data)
        else:
            rotated_compressed_data = self.rotate_and_compress(data_to_cc=self.unaligned_data)
        rotated_compressed_data = torch.flip(rotated_compressed_data, dims=[1])

        return fft2(
            torch.view_as_real(rotated_compressed_data.permute(2, 0, 1)),
            self.fft_centered,
            self.fft_normalization,
            self.spatial_dims,
        )

    def crop(
        self,
    ):
        """Crop to the size of the calibration lines."""
        s = torch.as_tensor([self.calib_lines, self.width, self.coils])

        idx = [
            torch.arange(
                abs(int(self.data.shape[n] // 2 + torch.ceil(-s[n] / 2))),
                abs(int(self.data.shape[n] // 2 + torch.ceil(s[n] / 2) + 1)),
            )
            for n in range(len(s))
        ]

        self.data = (
            self.data[idx[0][0] : idx[0][-1], idx[1][0] : idx[1][-1], idx[2][0] : idx[2][-1]]
            .unsqueeze(-2)
            .permute(1, 0, 2, 3)
        )

    def calculate_gcc(self):
        """Calculates Geometric Coil-Compression."""
        ws = (self.virtual_coils // 2) * 2 + 1

        Nx, Ny, Nz, Nc = self.data.shape

        im = torch.view_as_complex(
            ifft2(torch.view_as_real(self.data), self.fft_centered, self.fft_normalization, spatial_dims=0)
        )

        s = torch.as_tensor([Nx + ws - 1, Ny, Nz, Nc])
        idx = [
            torch.arange(
                abs(int(im.shape[n] // 2 + torch.ceil((-s[n] / 2).clone().detach()))),
                abs(int(im.shape[n] // 2 + torch.ceil((s[n] / 2).clone().detach())) + 1),
            )
            for n in range(len(s))
        ]

        zpim = torch.zeros((Nx + ws - 1, Ny, Nz, Nc)).type(im.dtype)
        zpim[idx[0][0] : idx[0][-1], idx[1][0] : idx[1][-1], idx[2][0] : idx[2][-1], idx[3][0] : idx[3][-1]] = im

        self.unaligned_data = torch.zeros((Nc, min(Nc, ws * Ny * Nz), Nx)).type(im.dtype)
        for n in range(Nx):
            tmpc = reshape_fortran(zpim[n : n + ws, :, :, :], (ws * Ny * Nz, Nc))
            _, _, v = torch.svd(tmpc, some=False)
            self.unaligned_data[:, :, n] = v

        self.unaligned_data = self.unaligned_data[:, : self.virtual_coils, :]

    def align_compressed_coils(self):
        """Virtual Coil Alignment."""
        self.aligned_data = self.unaligned_data

        _, sy, nc = self.aligned_data.shape
        ncc = sy

        n0 = nc // 2

        A00 = self.aligned_data[:, :ncc, n0 - 1]

        A0 = A00
        for n in range(n0, 0, -1):
            A1 = self.aligned_data[:, :ncc, n - 1]
            C = torch.conj(A1).T @ A0
            u, _, v = torch.svd(C, some=False)
            P = v @ torch.conj(u).T
            self.aligned_data[:, :ncc, n - 1] = A1 @ torch.conj(P).T
            A0 = self.aligned_data[:, :ncc, n - 1]

        A0 = A00
        for n in range(n0, nc):
            A1 = self.aligned_data[:, :ncc, n]
            C = torch.conj(A1).T @ A0
            u, _, v = torch.svd(C, some=False)
            P = v @ torch.conj(u).T
            self.aligned_data[:, :ncc, n] = A1 @ torch.conj(P).T
            A0 = self.aligned_data[:, :ncc, n]

    def rotate_and_compress(self, data_to_cc):
        """Uses compression matrices to project the data onto them -> rotate to the compressed space."""
        _data = self.init_data.permute(1, 0, 2).unsqueeze(-2)
        _ncc = data_to_cc.shape[1]

        data_to_cc = data_to_cc.to(_data.device)

        Nx, Ny, Nz, Nc = _data.shape
        im = torch.view_as_complex(
            ifft2(torch.view_as_real(_data), self.fft_centered, self.fft_normalization, spatial_dims=0)
        )

        ccdata = torch.zeros((Nx, Ny, Nz, _ncc)).type(_data.dtype).to(_data.device)
        for n in range(Nx):
            tmpc = im[n, :, :, :].squeeze().reshape(Ny * Nz, Nc)
            ccdata[n, :, :, :] = (tmpc @ data_to_cc[:, :, n]).reshape(Ny, Nz, _ncc).unsqueeze(0)

        ccdata = (
            torch.view_as_complex(
                fft2(torch.view_as_real(ccdata), self.fft_centered, self.fft_normalization, spatial_dims=0)
            )
            .permute(1, 0, 2, 3)
            .squeeze()
        )

        # Singlecoil
        if ccdata.dim() == 2:
            ccdata = ccdata.unsqueeze(-1)

        gcc = torch.zeros(ccdata.shape).type(ccdata.dtype)
        for n in range(ccdata.shape[-1]):
            gcc[:, :, n] = torch.view_as_complex(
                ifft2(
                    torch.view_as_real(ccdata[:, :, n]), self.fft_centered, self.fft_normalization, self.spatial_dims
                )
            )

        return gcc
