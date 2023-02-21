# coding=utf-8
from __future__ import annotations

from math import sqrt
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from omegaconf import DictConfig

import mridc.collections.reconstruction.nn as reconstruction_nn
from mridc.collections.common.data import subsample
from mridc.collections.common.parts import fft as fft
from mridc.collections.common.parts import utils as utils

__all__ = [
    "NoisePreWhitening",
    "GeometricDecompositionCoilCompression",
    "ZeroFilling",
    "Cropper",
    "Masker",
    "Composer",
    "MRIDataTransforms",
]


class NoisePreWhitening:
    """
    Applies noise pre-whitening / coil decorrelation.

    Parameters
    ----------
    find_patch_size : bool
        Find optimal patch size (automatically) to calculate psi. If False, patch_size must be defined.
        Default is ``True``.
    patch_size : list of ints
        Define patch size to calculate psi, [x_start, x_end, y_start, y_end].
    scale_factor : float
        Applied on the noise covariance matrix. Used to adjust for effective noise bandwidth and difference in
        sampling rate between noise calibration and actual measurement.
        scale_factor = (T_acq_dwell/T_noise_dwell)*NoiseReceiverBandwidthRatio
        Default is ``1.0``.
    fft_centered : bool
        If True, the zero-frequency component is located at the center of the spectrum.
        Default is ``False``.
    fft_normalization : str
        Normalization mode. Options are ``"backward"``, ``"ortho"``, ``"forward"``.
        Default is ``"backward"``.
    spatial_dims : sequence of ints
        Spatial dimensions of the input data.

    Examples
    --------
    >>> import torch
    >>> from mridc.collections.common.parts.transforms import NoisePreWhitening
    >>> data = torch.randn([30, 100, 100], dtype=torch.complex64)
    >>> data = torch.view_as_real(data)
    >>> data.mean()
    tensor(-0.0011)
    >>> noise_prewhitening = NoisePreWhitening(find_patch_size=True, scale_factor=1.0)
    >>> noise_prewhitening(data).mean()
    tensor(-0.0023)
    """

    def __init__(
        self,
        find_patch_size: bool = True,
        patch_size: List[int] = None,
        scale_factor: float = 1.0,
        fft_centered: bool = False,
        fft_normalization: str = "backward",
        spatial_dims: Sequence[int] = (-2, -1),
    ):
        super().__init__()
        # TODO: account for multiple echo times
        self.find_patch_size = find_patch_size
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.fft_centered = fft_centered
        self.fft_normalization = fft_normalization
        self.spatial_dims = spatial_dims

    def __call__(
        self,
        data: torch.Tensor,
        apply_backward_transform: bool = False,
        apply_forward_transform: bool = False,
    ) -> torch.Tensor:
        return self.forward(data, apply_backward_transform, apply_forward_transform)

    def __repr__(self):
        return f"Noise pre-whitening is applied with patch size {self.patch_size}."

    def __str__(self):
        return str(self.__repr__)

    def forward(
        self,
        data: torch.Tensor,
        apply_backward_transform: bool = False,
        apply_forward_transform: bool = False,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        data : torch.Tensor
            Input data to apply noise pre-whitening.
        apply_backward_transform : bool
            Apply backward transform before noise pre-whitening.
        apply_forward_transform : bool
            Apply forward transform before noise pre-whitening.

        Returns
        -------
        torch.Tensor
            Noise pre-whitened data.
        """
        if apply_forward_transform:
            data = fft.fft2(
                data,
                centered=self.fft_centered,
                normalization=self.fft_normalization,
                spatial_dims=self.spatial_dims,
            )

        if not self.patch_size:
            raise ValueError("Patch size must be defined for noise prewhitening.")

        if data.shape[-1] != 2:
            data = torch.view_as_real(data)

        if self.find_patch_size:
            patch = self.find_optimal_patch_size(data)
            noise = data[:, patch[0] : patch[1], patch[2] : patch[3]]
        elif not utils.is_none(self.patch_size):
            noise = data[:, self.patch_size[0] : self.patch_size[1], self.patch_size[-2] : self.patch_size[-1]]
        else:
            raise ValueError(
                "No patch size has been defined, while find_patch_size is False for noise prewhitening."
                "Please define a patch size or set find_patch_size to True."
            )
        noise_int = torch.reshape(noise, (noise.shape[0], int(torch.numel(noise) / noise.shape[0])))

        deformation_matrix = (1 / (float(noise_int.shape[1]) - 1)) * torch.mm(noise_int, torch.conj(noise_int).t())
        # ensure that the matrix is positive definite
        deformation_matrix = deformation_matrix + torch.eye(deformation_matrix.shape[0]) * 1e-6
        psi = torch.linalg.inv(torch.linalg.cholesky(deformation_matrix)) * sqrt(2) * sqrt(self.scale_factor)

        data = torch.reshape(
            torch.mm(psi, torch.reshape(data, (data.shape[0], int(torch.numel(data) / data.shape[0])))), data.shape
        )

        if apply_forward_transform:
            data = fft.ifft2(
                data,
                centered=self.fft_centered,
                normalization=self.fft_normalization,
                spatial_dims=self.spatial_dims,
            )

        return data.detach().clone()

    @staticmethod
    def find_optimal_patch_size(data: torch.Tensor, min_noise: float = 1e10) -> List[int]:
        """
        Find optimal patch size for noise pre-whitening.

        Parameters
        ----------
        data : torch.Tensor
            Input data to find optimal patch size.
        min_noise : float
            Minimum noise value. It is inversely proportional to the noise level. Default is ``1e10``.

        Returns
        -------
        List[int]
            Optimal patch size, [x_start, x_end, y_start, y_end].
        """
        if data.shape[-1] == 2:
            data = torch.view_as_complex(data)
        best_patch = []
        for patch_length in [10, 20, 30, 40, 50]:
            for patch_start_x in range(0, data.shape[-2] - patch_length, 10):
                for patch_start_y in range(0, data.shape[-1] - patch_length, 10):
                    patch = torch.abs(
                        utils.rss(
                            data[
                                :,
                                patch_start_x : patch_start_x + patch_length,
                                patch_start_y : patch_start_y + patch_length,
                            ],
                        )
                    )
                    noise = torch.sqrt(
                        torch.sum(torch.abs(patch - torch.mean(patch)) ** 2) / (len(torch.flatten(patch)) - 1)
                    )
                    if noise < min_noise:
                        min_noise = noise
                        best_patch = [
                            patch_start_x,
                            patch_start_x + patch_length,
                            patch_start_y,
                            patch_start_y + patch_length,
                        ]
        return best_patch


class GeometricDecompositionCoilCompression:
    """
    Geometric Decomposition Coil Compression in PyTorch, as presented in [1].

    References
    ----------
    .. [1] Zhang, T., Pauly, J. M., Vasanawala, S. S., & Lustig, M. (2013). Coil compression for accelerated imaging
        with Cartesian sampling. Magnetic Resonance in Medicine, 69(2), 571–582. https://doi.org/10.1002/mrm.24267

    Parameters
    ----------
    virtual_coils : int
        Number of final-"virtual" coils.
    calib_lines : int
        Calibration lines to sample data points.
    align_data : bool
        Align data to the first calibration line. Default is ``True``.
    fft_centered : bool
        Whether to center the fft. Default is ``False``.
    fft_normalization : str
        FFT normalization. Default is ``"backward"``.
    spatial_dims : Sequence[int]
        Dimensions to apply the FFT. Default is ``None``.

    Returns
    -------
    torch.Tensor
        Coil compressed data.

    Examples
    --------
    >>> import torch
    >>> from mridc.collections.common.parts.transforms import GeometricDecompositionCoilCompression
    >>> data = torch.randn([30, 100, 100], dtype=torch.complex64)
    >>> gdcc = GeometricDecompositionCoilCompression(virtual_coils=10, calib_lines=24, spatial_dims=[-2, -1])
    >>> gdcc(data).shape
    torch.Size([10, 100, 100, 2])
    """

    def __init__(
        self,
        virtual_coils: int = None,
        calib_lines: int = None,
        align_data: bool = True,
        fft_centered: bool = False,
        fft_normalization: str = "backward",
        spatial_dims: Sequence[int] = (-2, -1),
    ):
        super().__init__()
        # TODO: account for multiple echo times
        self.virtual_coils = virtual_coils
        self.calib_lines = calib_lines
        self.align_data = align_data
        self.fft_centered = fft_centered
        self.fft_normalization = fft_normalization
        self.spatial_dims = spatial_dims

    def __call__(
        self,
        data: Union[torch.Tensor, None],
        apply_backward_transform: bool = False,
        apply_forward_transform: bool = False,
    ) -> torch.Tensor:
        if not utils.is_none(data) and data.dim() > 1 and data.mean() != 1:  # type: ignore
            return self.forward(data, apply_backward_transform, apply_forward_transform)
        return data

    def __repr__(self):
        return f"Coil Compression is applied reducing coils to {self.virtual_coils}."

    def __str__(self):
        return str(self.__repr__)

    def forward(
        self,
        data: torch.Tensor,
        apply_backward_transform: bool = False,
        apply_forward_transform: bool = False,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        data : torch.Tensor
            Input data to apply coil compression.
        apply_backward_transform : bool
            Apply backward transform. Default is ``False``.
        apply_forward_transform : bool
            Apply forward transform. Default is ``False``.

        Returns
        -------
        torch.Tensor
            Coil compressed data.
        """
        if not self.virtual_coils:
            raise ValueError("Number of virtual coils must be defined for geometric decomposition coil compression.")

        if apply_forward_transform:
            data = fft.fft2(
                data,
                centered=self.fft_centered,
                normalization=self.fft_normalization,
                spatial_dims=self.spatial_dims,
            )

        self.data = data
        if self.data.shape[-1] == 2:
            self.data = torch.view_as_complex(self.data)

        curr_num_coils = self.data.shape[0]
        if curr_num_coils < self.virtual_coils:
            raise ValueError(
                f"Tried to compress from {curr_num_coils} to {self.virtual_coils} coils, please select less coils."
            )

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
        rotated_compressed_data = torch.view_as_real(rotated_compressed_data.permute(2, 0, 1))

        if not apply_forward_transform:
            rotated_compressed_data = fft.fft2(
                rotated_compressed_data,
                self.fft_centered,
                self.fft_normalization,
                self.spatial_dims,
            )

        return rotated_compressed_data.detach().clone()

    def crop(self):
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
            fft.ifft2(torch.view_as_real(self.data), self.fft_centered, self.fft_normalization, spatial_dims=0)
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
            tmpc = utils.reshape_fortran(zpim[n : n + ws, :, :, :], (ws * Ny * Nz, Nc))
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
            fft.ifft2(torch.view_as_real(_data), self.fft_centered, self.fft_normalization, spatial_dims=0)
        )

        ccdata = torch.zeros((Nx, Ny, Nz, _ncc)).type(_data.dtype).to(_data.device)
        for n in range(Nx):
            tmpc = im[n, :, :, :].squeeze().reshape(Ny * Nz, Nc)
            ccdata[n, :, :, :] = (tmpc @ data_to_cc[:, :, n]).reshape(Ny, Nz, _ncc).unsqueeze(0)

        ccdata = (
            torch.view_as_complex(
                fft.fft2(torch.view_as_real(ccdata), self.fft_centered, self.fft_normalization, spatial_dims=0)
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
                fft.ifft2(
                    torch.view_as_real(ccdata[:, :, n]), self.fft_centered, self.fft_normalization, self.spatial_dims
                )
            )

        return gcc


class ZeroFilling:
    """
    Zero-Filling transform.

    Parameters
    ----------
    zero_filling_size : tuple
        Size of the zero filled data.

    Returns
    -------
    zero_filled_data : torch.Tensor
        Zero filled data.
    spatial_dims : tuple
        Spatial dimensions.

    Example
    -------
    >>> import torch
    >>> from mridc.collections.common.parts.transforms import ZeroFilling
    >>> data = torch.randn(1, 15, 320, 320, 2)
    >>> zero_filling = ZeroFilling(zero_filling_size=(400, 400), spatial_dims=(-2, -1)) # don't account for complex dim
    >>> zero_filled_data = zero_filling(data)
    >>> zero_filled_data.shape
    [1, 15, 400, 400, 2]
    """

    def __init__(
        self,
        zero_filling_size: Tuple,
        fft_centered: bool = False,
        fft_normalization: str = "backward",
        spatial_dims: Sequence[int] = (-2, -1),
    ):
        self.zero_filling_size = zero_filling_size
        self.fft_centered = fft_centered
        self.fft_normalization = fft_normalization
        self.spatial_dims = spatial_dims

    def __call__(
        self,
        data: Union[torch.Tensor, None],
        apply_backward_transform: bool = False,
        apply_forward_transform: bool = False,
    ) -> torch.Tensor:
        """Applies zero filling to data."""
        if not utils.is_none(data) and data.dim() > 1 and data.mean() != 1:  # type: ignore
            return self.forward(data, apply_backward_transform, apply_forward_transform)
        return data

    def __repr__(self) -> str:
        return f"Zero-Filling will be applied to data with size {self.zero_filling_size}."

    def __str__(self) -> str:
        return self.__repr__()

    def forward(
        self,
        data: torch.Tensor,
        apply_backward_transform: bool = False,
        apply_forward_transform: bool = False,
    ) -> torch.Tensor:
        """Applies zero filling to data."""
        if apply_backward_transform:
            data = fft.ifft2(
                data,
                centered=self.fft_centered,
                normalization=self.fft_normalization,
                spatial_dims=self.spatial_dims,
            )
        elif apply_forward_transform:
            data = fft.fft2(
                data,
                centered=self.fft_centered,
                normalization=self.fft_normalization,
                spatial_dims=self.spatial_dims,
            )

        is_complex = data.shape[-1] == 2

        if is_complex:
            data = torch.view_as_complex(data)

        padding_top = np.floor_divide(abs(int(self.zero_filling_size[0]) - data.shape[self.spatial_dims[0]]), 2)
        padding_bottom = padding_top
        padding_left = np.floor_divide(abs(int(self.zero_filling_size[1]) - data.shape[self.spatial_dims[1]]), 2)
        padding_right = padding_left

        data = torch.nn.functional.pad(
            data, pad=(padding_left, padding_right, padding_top, padding_bottom), mode="constant", value=0
        )

        if is_complex:
            data = torch.view_as_real(data)

        if apply_backward_transform:
            data = fft.fft2(
                data,
                centered=self.fft_centered,
                normalization=self.fft_normalization,
                spatial_dims=self.spatial_dims,
            )
        elif apply_forward_transform:
            data = fft.ifft2(
                data,
                centered=self.fft_centered,
                normalization=self.fft_normalization,
                spatial_dims=self.spatial_dims,
            )

        return data


class Cropper:
    """
    Cropper transform.

    Parameters
    ----------
    cropping_size : tuple
        Size of the cropped data.
    fft_centered : bool
        If True, the input is assumed to be centered in the frequency domain. Default is `False`.
    fft_normalization : str
        Normalization of the FFT. Default is `backward`.
    spatial_dims : tuple
        Spatial dimensions.

    Returns
    -------
    cropped_data : torch.Tensor
        Cropped data.

    Example
    -------
    >>> import torch
    >>> from mridc.collections.common.parts.transforms import Cropper
    >>> data = torch.randn(1, 15, 320, 320, 2)
    >>> cropping = Cropper(cropping_size=(256, 256), spatial_dims=(-2, -1)) # don't account for complex dim
    >>> cropped_data = cropping(data)
    >>> cropped_data.shape
    [1, 15, 256, 256, 2]
    """

    def __init__(
        self,
        cropping_size: Tuple,
        fft_centered: bool = False,
        fft_normalization: str = "backward",
        spatial_dims: Sequence[int] = (-2, -1),
    ):
        self.cropping_size = cropping_size
        self.fft_centered = fft_centered
        self.fft_normalization = fft_normalization
        self.spatial_dims = spatial_dims

    def __call__(
        self,
        data: Union[torch.Tensor, List[torch.Tensor], None],
        apply_backward_transform: bool = False,
        apply_forward_transform: bool = False,
    ) -> List[torch.Tensor] | torch.Tensor:
        if not utils.is_none(data):  # type: ignore
            if isinstance(data, list) and len(data) > 0:  # type: ignore
                return [self.forward(d, apply_backward_transform, apply_forward_transform) for d in data]
            elif data.dim() > 1 and data.mean() != 1:  # type: ignore
                return self.forward(data, apply_backward_transform, apply_forward_transform)
        return data

    def __repr__(self):
        return f"Data will be cropped to size={self.cropping_size}."

    def __str__(self):
        return self.__repr__()

    def forward(
        self,
        data: torch.Tensor,
        apply_backward_transform: bool = False,
        apply_forward_transform: bool = False,
    ) -> torch.Tensor:
        """Applies cropping to data."""
        if apply_backward_transform:
            data = fft.ifft2(
                data,
                centered=self.fft_centered,
                normalization=self.fft_normalization,
                spatial_dims=self.spatial_dims,
            )
        elif apply_forward_transform:
            data = fft.fft2(
                data,
                centered=self.fft_centered,
                normalization=self.fft_normalization,
                spatial_dims=self.spatial_dims,
            )

        is_complex = data.shape[-1] == 2

        if is_complex:
            data = torch.view_as_complex(data)

        crop_size = (data.shape[self.spatial_dims[0]], data.shape[self.spatial_dims[1]])

        # Check for smallest size against the target shape.
        h = min(int(self.cropping_size[0]), crop_size[0])
        w = min(int(self.cropping_size[1]), crop_size[1])

        # Check for smallest size against the stored recon shape in data.
        if crop_size[0] != 0:
            h = h if h <= crop_size[0] else crop_size[0]
        if crop_size[1] != 0:
            w = w if w <= crop_size[1] else crop_size[1]

        self.cropping_size = (int(h), int(w))

        data = utils.center_crop(data, self.cropping_size)

        if is_complex:
            data = torch.view_as_real(data)

        if apply_backward_transform:
            data = fft.fft2(
                data,
                centered=self.fft_centered,
                normalization=self.fft_normalization,
                spatial_dims=self.spatial_dims,
            )
        elif apply_forward_transform:
            data = fft.ifft2(
                data,
                centered=self.fft_centered,
                normalization=self.fft_normalization,
                spatial_dims=self.spatial_dims,
            )

        return data


class Masker:
    """
    Masker transform.

    Parameters
    ----------
    mask_func : callable, optional
        Masker function. Default is `None`.
    spatial_dims : tuple
        Spatial dimensions. Default is `(-2, -1)`.
    shift_mask : bool
        Whether to shift the mask. Default is `False`.
    half_scan_percentage : float
        Whether to simulate half scan. Default is `0.0`, which means no half scan.
    center_scale : float
        Percentage of center to remain densely sampled. Default is `0.02`.
    dimensionality : int
        Dimensionality of the data. Default is `2`.
    remask: bool
        Whether to remask the data. If False, the mask will be generated only once. If True, the mask will be generated
        every time the transform is called. Default is `False`.

    Returns
    -------
    Tuple[List[torch.Tensor], List[torch.Tensor], List[float]]
        Masked data, mask, and acceleration factor. They are returned as a tuple of lists, where each list corresponds
        to a different acceleration factor. If one acceleration factor is provided, the lists will be of length 1.

    Example
    -------
    >>> import torch
    >>> from mridc.collections.common.parts.transforms import Masker
    >>> data = torch.randn(1, 15, 320, 320, 2)
    >>> mask = torch.ones(320, 320)
    >>> masker = Masker(mask_func=None, spatial_dims=(-2, -1), shift_mask=False, half_scan_percentage=0.0, \
    center_scale=0.02, dimensionality=2, remask=True)
    >>> masked_data = masker(data, mask, seed=None)
    >>> masked_data[0][0].shape  # masked data
    [1, 15, 320, 320, 2]
    >>> masked_data[1][0].shape  # mask
    [320, 320]
    >>> masked_data[2][0]  # acceleration factor
    10.0
    """

    def __init__(
        self,
        mask_func: Optional[Callable] = None,
        spatial_dims: Sequence[int] = (-2, -1),
        shift_mask: bool = False,
        half_scan_percentage: float = 0.0,
        center_scale: float = 0.02,
        dimensionality: int = 2,
        remask: bool = True,
    ):
        self.mask_func = mask_func
        self.spatial_dims = spatial_dims
        self.shift_mask = shift_mask
        self.half_scan_percentage = half_scan_percentage
        self.center_scale = center_scale
        self.dimensionality = dimensionality
        self.remask = remask

    def __call__(
        self,
        data: torch.Tensor,
        mask: Union[List, torch.Tensor, np.ndarray] = None,
        padding: Optional[Tuple] = None,
        seed: Optional[int] = None,
        apply_backward_transform: bool = False,
        apply_forward_transform: bool = False,
    ) -> Tuple[
        List[float | torch.Tensor | Any],
        List[torch.Tensor | Any] | List[torch.Tensor | np.ndarray | None | Any],
        List[int | torch.Tensor | Any],
    ]:
        """Applies mask to data."""
        # Check if mask is precomputed or not.
        if not utils.is_none(mask):
            if isinstance(mask, list):
                if len(mask) == 0:
                    mask = None
            elif mask.ndim() == 0:  # type: ignore
                mask = None

        if not utils.is_none(mask) and isinstance(mask, list) and len(mask) > 0:
            self.__type__ = "Masks are precomputed and loaded."
        elif (
            not utils.is_none(mask)
            and not isinstance(mask, list)
            and mask.ndim != 0  # type: ignore
            and len(mask) > 0  # type: ignore
        ):
            self.__type__ = "Mask is either precomputed and loaded or data are prospectively undersampled."
        elif isinstance(self.mask_func, list):
            self.__type__ = "A number accelerations are provided and masks are generated on the fly."
        else:
            self.__type__ = "A single acceleration is provided and mask is generated on the fly."
        return self.forward(data, mask, padding, seed)

    def __repr__(self) -> str:
        return f"{self.__type__}"

    def __str__(self) -> str:
        return self.__repr__()

    def forward(
        self,
        data: torch.Tensor,
        mask: Union[List, torch.Tensor, np.ndarray] = None,
        padding: Optional[Tuple] = None,
        seed: Optional[int] = None,
    ) -> Tuple[
        List[float | torch.Tensor | Any],
        List[torch.Tensor | Any] | List[torch.Tensor | np.ndarray | None | Any],
        List[int | torch.Tensor | Any],
    ]:
        """Function to apply mask to data."""
        is_complex = data.shape[-1] == 2

        if is_complex:
            self.spatial_dims = tuple([x - 1 for x in self.spatial_dims])

        if not utils.is_none(mask) and isinstance(mask, list) and len(mask) > 0:
            masked_data = []
            masks = []
            accelerations = []
            for i, m in enumerate(mask):
                if list(m.shape) == [data.shape[self.spatial_dims[0]], data.shape[self.spatial_dims[1]]]:
                    if isinstance(m, np.ndarray):
                        m = torch.from_numpy(m)
                    m = m.unsqueeze(0).unsqueeze(-1)

                if not utils.is_none(padding[0]) and padding[0] != 0:  # type: ignore
                    m[:, :, : padding[0]] = 0  # type: ignore
                    m[:, :, padding[1] :] = 0  # type: ignore

                if self.shift_mask:
                    m = torch.fft.fftshift(m, dim=(self.spatial_dims[0], self.spatial_dims[1]))

                masked_data.append(data * m + 0.0)
                masks.append(m)
                accelerations.append(m.sum() / m.numel())

        elif (
            not utils.is_none(mask)
            and not isinstance(mask, list)
            and mask.ndim != 0  # type: ignore
            and len(mask) > 0  # type: ignore
        ):
            if list(mask.shape) == [  # type: ignore
                data.shape[self.spatial_dims[0]],
                data.shape[self.spatial_dims[1]],
            ]:  # type: ignore
                if isinstance(mask, np.ndarray):
                    mask = torch.from_numpy(mask)
                mask = mask.unsqueeze(0).unsqueeze(-1)

            if not utils.is_none(padding[0]) and padding[0] != 0:  # type: ignore
                mask[:, :, : padding[0]] = 0  # type: ignore
                mask[:, :, padding[1] :] = 0  # type: ignore

            if self.shift_mask:
                mask = torch.fft.fftshift(mask, dim=(self.spatial_dims[0], self.spatial_dims[1]))

            masked_data = [data * mask + 0.0]
            masks = [mask]
            accelerations = [mask.sum() / mask.numel()]

        elif isinstance(self.mask_func, list):
            masked_data = []
            masks = []
            accelerations = []
            for i, m in enumerate(self.mask_func):
                if self.dimensionality == 2:
                    _masked_data, _mask, _accelerations = utils.apply_mask(
                        data,
                        m,
                        seed,
                        padding,
                        shift=self.shift_mask,
                        half_scan_percentage=self.half_scan_percentage,
                        center_scale=self.center_scale,
                    )
                elif self.dimensionality == 3:
                    _masked_data = []
                    _masks = []
                    _accelerations = []
                    j_mask = None
                    for j in range(data.shape[0]):
                        j_masked_data, j_mask, j_acc = utils.apply_mask(
                            data[j],
                            m,
                            seed,
                            padding,
                            shift=self.shift_mask,
                            half_scan_percentage=self.half_scan_percentage,
                            center_scale=self.center_scale,
                            existing_mask=j_mask if not self.remask else None,
                        )
                        _masked_data.append(j_masked_data)
                        _masks.append(j_mask)
                        _accelerations.append(j_acc)
                    _masked_data = torch.stack(_masked_data, dim=0)
                    _mask = torch.stack(_masks, dim=0)
                    _accelerations = torch.stack(_accelerations, dim=0)  # type: ignore
                else:
                    raise ValueError(f"Unsupported data dimensionality {self.dimensionality}D.")
                masked_data.append(_masked_data)
                masks.append(_mask)
                accelerations.append(_accelerations)

        elif not utils.is_none(self.mask_func):
            masked_data, masks, accelerations = utils.apply_mask(  # type: ignore
                data,
                self.mask_func[0],  # type: ignore
                seed,
                padding,
                shift=self.shift_mask,
                half_scan_percentage=self.half_scan_percentage,
                center_scale=self.center_scale,
            )
            masked_data = [masked_data]
            masks = [masks]
            accelerations = [accelerations]

        else:
            masked_data = [data]
            masks = [torch.empty([])]
            accelerations = [torch.empty([])]

        return masked_data, masks, accelerations


class Normalizer:
    """
    Normalizes data given a normalization type.

    Parameters
    ----------
    normalization_type: str, optional
        Normalization type. It can be one of the following:
        - "max": normalize data by its maximum value.
        - "mean": normalize data by its mean value.
        - "minmax": normalize data by its minimum and maximum values.
        - None: do not normalize data. It can be useful to verify FFT normalization.
        Default is `None`.
    fft_centered: bool, optional
        If True, the FFT will be centered. Default is `False`. Should be set for complex data normalization.
    fft_normalization: str, optional
        FFT normalization type. It can be one of the following:
        - "backward": normalize the FFT by the number of elements in the input.
        - "ortho": normalize the FFT by the number of elements in the input and the square root of the product of the
        sizes of the input dimensions.
        - "forward": normalize the FFT by the square root of the number of elements in the input.
        Default is "backward".
    spatial_dims: tuple, optional
        Spatial dimensions. Default is `(-2, -1)`.

    Returns
    -------
    normalized_data: torch.Tensor
        Normalized data to range according to the normalization type.

    Example
    --------
    >>> import torch
    >>> from mridc.collections.common.parts.transforms import Normalizer
    >>> data = torch.randn(2, 2, 2, 2, 2) + 1j * torch.randn(2, 2, 2, 2, 2)
    >>> print(torch.min(torch.abs(data)), torch.max(torch.abs(data)))
    tensor(1e-06) tensor(1.4142)
    >>> normalizer = Normalizer(normalization_type="max")
    >>> normalized_data = normalizer(data)
    >>> print(torch.min(torch.abs(data)), torch.max(torch.abs(data)))
    tensor(0.) tensor(1.)
    """

    def __init__(
        self,
        normalization_type: Optional[str] = None,
        fft_centered: bool = False,
        fft_normalization: str = "backward",
        spatial_dims: Sequence[int] = (-2, -1),
    ):
        self.normalization_type = normalization_type
        self.fft_centered = fft_centered
        self.fft_normalization = fft_normalization
        self.spatial_dims = spatial_dims

    def __call__(
        self,
        data: Union[torch.Tensor, List[torch.Tensor], None],
        apply_backward_transform: bool = False,
        apply_forward_transform: bool = False,
    ) -> List[torch.Tensor] | torch.Tensor:
        if not utils.is_none(data):
            if isinstance(data, list) and len(data) > 0:
                return [self.forward(d, apply_backward_transform, apply_forward_transform) for d in data]
            elif data.dim() > 1 and data.mean() != 1:  # type: ignore
                return self.forward(data, apply_backward_transform, apply_forward_transform)
        return data

    def __repr__(self):
        return f"Normalization type is set to {self.normalization_type}."

    def __str__(self):
        return self.__repr__()

    def forward(
        self,
        data: torch.Tensor,
        apply_backward_transform: bool = False,
        apply_forward_transform: bool = False,
    ) -> torch.Tensor:
        if apply_backward_transform:
            data = fft.ifft2(
                data,
                centered=self.fft_centered,
                normalization=self.fft_normalization,
                spatial_dims=self.spatial_dims,
            )
        elif apply_forward_transform:
            data = fft.fft2(
                data,
                centered=self.fft_centered,
                normalization=self.fft_normalization,
                spatial_dims=self.spatial_dims,
            )

        if self.normalization_type == "max":
            data = data / torch.max(torch.abs(data))
        elif self.normalization_type == "mean":
            data = data - torch.mean(torch.abs(data))
            data = data / torch.std(torch.abs(data))
        elif self.normalization_type == "minmax":
            min_value = torch.min(torch.abs(data))
            data = (data - min_value) / (torch.max(torch.abs(data)) - min_value)

        if apply_backward_transform:
            data = fft.fft2(
                data,
                centered=self.fft_centered,
                normalization=self.fft_normalization,
                spatial_dims=self.spatial_dims,
            )
        elif apply_forward_transform:
            data = fft.ifft2(
                data,
                centered=self.fft_centered,
                normalization=self.fft_normalization,
                spatial_dims=self.spatial_dims,
            )

        return data


class Composer:
    """
    Composes multiple transforms together.

    Parameters
    ----------
    transforms: list
        List of transforms to compose.

    Returns
    -------
    composed_data: torch.Tensor
        Composed data.

    Example
    --------
    >>> import torch
    >>> from mridc.collections.common.parts.transforms import Composer, Masker, Normalizer
    >>> data = torch.randn(2, 2, 2, 2, 2) + 1j * torch.randn(2, 2, 2, 2, 2)
    >>> print(torch.min(torch.abs(data)), torch.max(torch.abs(data)))
    tensor(1e-06) tensor(1.4142)
    >>> masker = Masker(mask_func="random", padding="reflection", seed=0)
    >>> normalizer = Normalizer(normalization_type="max")
    >>> composer = Composer([masker, normalizer])
    >>> composed_data = composer(data)
    >>> print(torch.min(torch.abs(composed_data)), torch.max(torch.abs(composed_data)))
    tensor(0.) tensor(1.)
    """

    def __init__(self, transforms: Union[List[Callable], Callable, None]):
        self.transforms = transforms

    def __call__(
        self,
        data: Union[torch.Tensor, List[torch.Tensor], None],
        apply_backward_transform: bool = False,
        apply_forward_transform: bool = False,
    ) -> List[torch.Tensor] | torch.Tensor:
        for transform in self.transforms:  # type: ignore
            if not utils.is_none(transform):
                data = transform(data, apply_backward_transform, apply_forward_transform)
        return data

    def __repr__(self):
        return f"Composed transforms: {self.transforms}"

    def __str__(self):
        return self.__repr__()


class MRIDataTransforms:
    """
    Generic class to apply transforms for MRI data.

    Parameters
    ----------
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
    MRIDataTransforms
        Preprocessed data.
    """

    def __init__(
        self,
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
        kspace, masked_kspace, mask, acc = self.__process_kspace__(kspace, mask, attrs, fname)
        sensitivity_map = self.__process_coil_sensitivities_map__(sensitivity_map, kspace)
        target = self.__initialize_prediction__(target, kspace, sensitivity_map)
        prediction = self.__initialize_prediction__(prediction, masked_kspace, sensitivity_map)
        return kspace, masked_kspace, sensitivity_map, mask, prediction, target, fname, slice_idx, acc

    def __repr__(self) -> str:
        return (
            f"Preprocessing transforms initialized for {self.__class__.__name__}: "
            f"prewhitening={self.prewhitening}, "
            f"masking={self.masking}, "
            f"kspace_zero_filling={self.kspace_zero_filling}, "
            f"cropping={self.cropping}, "
            f"normalization={self.normalization}, "
            f"init_reconstructor={self.init_reconstructor}, "
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
        self, prediction: np.ndarray, kspace: torch.Tensor, sensitivity_map: torch.Tensor
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
        if utils.is_none(prediction) or prediction.ndim < 2:
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
        return prediction
