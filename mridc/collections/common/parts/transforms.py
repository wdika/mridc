# coding=utf-8
from __future__ import annotations

import os
import time
from math import sqrt
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor

import mridc.collections.reconstruction.nn as reconstruction_nn
from mridc.collections.common.data import subsample
from mridc.collections.common.parts import fft, utils

__all__ = [
    "Composer",
    "Cropper",
    "GeometricDecompositionCoilCompression",
    "Masker",
    "MRIDataTransforms",
    "N2R",
    "NoisePreWhitening",
    "Normalizer",
    "SSDU",
    "ZeroFilling",
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

    def __init__(  # noqa: D107
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
        apply_backward_transform: bool = False,  # noqa: D102
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

    def __init__(  # noqa: D107
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
        apply_backward_transform: bool = False,  # noqa: D102
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
            if data.dim() > 1 and data.mean() != 1:  # type: ignore
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

    def __init__(  # noqa: C901
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

    def __call__(  # noqa: C901
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
            elif mask.ndim == 0:  # type: ignore
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

    def forward(  # noqa: C901
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
            spatial_dims = tuple([x - 1 for x in self.spatial_dims])  # noqa: WPS515
        else:
            spatial_dims = self.spatial_dims  # type: ignore

        if not utils.is_none(mask) and isinstance(mask, list) and len(mask) > 0:
            masked_data = []
            masks = []
            accelerations = []
            for m in mask:
                if list(m.shape) == [data.shape[spatial_dims[0]], data.shape[spatial_dims[1]]]:
                    if isinstance(m, np.ndarray):
                        m = torch.from_numpy(m)
                    m = m.unsqueeze(0).unsqueeze(-1)

                if not utils.is_none(padding[0]) and padding[0] != 0:  # type: ignore
                    m[:, :, : padding[0]] = 0  # type: ignore
                    m[:, :, padding[1] :] = 0  # type: ignore

                if self.shift_mask:
                    m = torch.fft.fftshift(m, dim=(spatial_dims[0], spatial_dims[1]))

                masked_data.append(data * m + 0.0)
                masks.append(m)
                accelerations.append(m.sum() / m.numel())

        elif (
            not utils.is_none(mask)
            and not isinstance(mask, list)
            and mask.ndim != 0  # type: ignore
            and len(mask) > 0  # type: ignore
        ):
            # if list(mask.shape) == [  # type: ignore
            #     data.shape[self.spatial_dims[0]],
            #     data.shape[self.spatial_dims[1]],
            # ]:  # type: ignore
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask)
            mask = mask.unsqueeze(0).unsqueeze(-1)

            if not utils.is_none(padding[0]) and padding[0] != 0:  # type: ignore
                mask[:, :, : padding[0]] = 0  # type: ignore
                mask[:, :, padding[1] :] = 0  # type: ignore

            if self.shift_mask:
                mask = torch.fft.fftshift(mask, dim=(spatial_dims[0], spatial_dims[1]))

            masked_data = [data * mask + 0.0]
            masks = [mask]
            accelerations = [mask.sum() / mask.numel()]

        elif isinstance(self.mask_func, list):
            masked_data = []
            masks = []
            accelerations = []
            for m in self.mask_func:
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


class SSDU:
    """
    Generates Self-Supervised Data Undersampling (SSDU) masks, as presented in [1].

    References
    ----------
    [1] Yaman, B, Hosseini, SAH, Moeller, S, Ellermann, J, Uğurbil, K, Akçakaya, M. Self-supervised learning of
        physics-guided reconstruction neural networks without fully sampled reference data. Magn Reson Med. 2020; 84:
        3172– 3191. https://doi.org/10.1002/mrm.28378

    Parameters
    ----------
    mask_type: str, optional
        Mask type. It can be one of the following:
        - "Gaussian": Gaussian sampling.
        - "Uniform": Uniform sampling.
        Default is "Gaussian".
    rho: float, optional
        Split ratio for training and loss masks. Default is ``0.4``.
    acs_block_size: Sequence[int], optional
        Keeps a small acs region fully-sampled for training masks, if there is no acs region. The small acs block
        should be set to zero. Default is ``(4, 4)``.
    gaussian_std_scaling_factor: float, optional
        Scaling factor for standard deviation of the Gaussian noise. If Uniform is select this factor is ignored.
        Default is ``4.0``.
    outer_kspace_fraction: float, optional
        Fraction of the outer k-space region to be kept/unmasked. Default is ``0.0``.
    export_and_reuse_masks: bool, optional
        If ``True``, the generated masks are exported to the tmp directory and reused for the next call. This
        option is useful when the data is too large to be stored in memory. Default is ``False``.

    Returns
    -------
    loss_mask: torch.Tensor
        Loss mask.
    training_mask: torch.Tensor
        Training mask.
    """

    def __init__(  # noqa: W0221
        self,
        mask_type: str = "Gaussian",
        rho: float = 0.4,
        acs_block_size: Sequence[int] = (4, 4),
        gaussian_std_scaling_factor: float = 4.0,
        outer_kspace_fraction: float = 0.0,
        export_and_reuse_masks: bool = False,
    ):
        if mask_type not in ["Gaussian", "Uniform"]:
            raise ValueError(f"SSDU mask type {mask_type} is not supported.")
        self.mask_type = mask_type
        self.rho = rho
        self.acs_block_size = acs_block_size
        self.gaussian_std_scaling_factor = gaussian_std_scaling_factor
        self.outer_kspace_fraction = outer_kspace_fraction
        self.export_and_reuse_masks = export_and_reuse_masks

    def __call__(self, data: torch.Tensor, mask: torch.Tensor, fname: str) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward(mask, fname)

    def __repr__(self):
        return f"SSDU type is set to {self.mask_type}."

    def __str__(self):
        return self.__repr__()

    def forward(self, mask: torch.Tensor, fname: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.export_and_reuse_masks:
            # check if masks are already generated
            precomputed_masks = self.__exists__(fname, (mask.shape[0], mask.shape[1]))
            if precomputed_masks is not None:
                return precomputed_masks[0], precomputed_masks[1]

        if self.mask_type == "Gaussian":
            _mask = self.__gaussian_sampling__(mask)
        else:
            _mask = self.__uniform_sampling__(mask)

        train_mask = torch.where(mask == 1, 1 - _mask, mask)  # type: ignore
        loss_mask = torch.where(mask == 1, _mask, mask)  # type: ignore

        # add the acs region to ensure linearity in FFT
        # train_mask = torch.where(self.__find_acs_region__(train_mask) == 1, 1, train_mask)  # type: ignore
        # loss_mask = torch.where(self.__find_acs_region__(mask) == 1, 1, loss_mask)  # type: ignore

        if self.outer_kspace_fraction > 0:
            train_mask = self.__apply_outer_kspace_unmask__(train_mask)
            loss_mask = self.__apply_outer_kspace_unmask__(loss_mask)

        if self.export_and_reuse_masks:
            # save masks
            self.__export__(torch.stack([train_mask, loss_mask], dim=0), fname)

        return train_mask, loss_mask

    @staticmethod
    def __find_acs_region__(mask: torch.Tensor) -> torch.Tensor:
        """
        Find the acs region.

        Parameters
        ----------
        mask : torch.Tensor
            Sampling mask.

        Returns
        -------
        torch.Tensor
            ACS region.
        """
        center = (mask.shape[0] // 2, mask.shape[1] // 2)

        # find the size of the acs region, start from the center and go left to find contiguous 1s
        acs_region = torch.zeros_like(mask)
        for i in range(center[0], 0, -1):
            if mask[i, center[1]] == 1:
                acs_region[i, :] = 1
            else:
                break

        # go right
        for i in range(center[0], mask.shape[0]):
            if mask[i, center[1]] == 1:
                acs_region[i, :] = 1
            else:
                break

        # go up
        for i in range(center[1], 0, -1):
            if mask[center[0], i] == 1:
                acs_region[:, i] = 1
            else:
                break

        # go down
        for i in range(center[1], mask.shape[1]):
            if mask[center[0], i] == 0:
                acs_region[:, i] = 1
            else:
                break

        # keep only the acs region
        # take only the first row and stop when you find a 1
        left = 0
        for i in range(acs_region.shape[0]):
            if acs_region[i, 0] == 1:
                left = i
                break

        # take only the last row and stop when you find a 1
        right = 0
        for i in range(acs_region.shape[0] - 1, 0, -1):
            if acs_region[i, 0] == 1:
                right = i
                break

        # take only the first column and stop when you find a 1
        up = 0
        for i in range(acs_region.shape[1]):
            if acs_region[0, i] == 1:
                up = i
                break

        # take only the last column and stop when you find a 1
        down = 0
        for i in range(acs_region.shape[1] - 1, 0, -1):
            if acs_region[0, i] == 1:
                down = i
                break

        acs_region = torch.zeros_like(mask)
        acs_region[left:right, up:down] = 1

        # keep only the part of the acs region that is in the mask
        return acs_region * mask

    def __gaussian_sampling__(self, mask: torch.Tensor) -> torch.Tensor:  # noqa: W0221
        nrow, ncol = mask.shape[0], mask.shape[1]
        center_kx = nrow // 2
        center_ky = ncol // 2

        tmp_mask = mask.clone()
        tmp_mask[
            center_kx - self.acs_block_size[0] // 2 : center_kx + self.acs_block_size[0] // 2,
            center_ky - self.acs_block_size[1] // 2 : center_ky + self.acs_block_size[1] // 2,
        ] = 0

        _mask = torch.zeros_like(mask)
        count = 0

        total = int(torch.ceil(torch.sum(mask[:]) * self.rho))

        while count <= total:
            indx = int(np.round(np.random.normal(loc=center_kx, scale=(nrow - 1) / self.gaussian_std_scaling_factor)))
            indy = int(np.round(np.random.normal(loc=center_ky, scale=(ncol - 1) / self.gaussian_std_scaling_factor)))

            if 0 <= indx < nrow and 0 <= indy < ncol and tmp_mask[indx, indy] == 1 and _mask[indx, indy] != 1:
                _mask[indx, indy] = 1
                count += 1

        return _mask

    def __uniform_sampling__(self, mask: torch.Tensor) -> torch.Tensor:  # noqa: W0221
        nrow, ncol = mask.shape[0], mask.shape[1]
        center_kx = nrow // 2
        center_ky = ncol // 2

        tmp_mask = mask.clone()
        tmp_mask[
            center_kx - self.acs_block_size[0] // 2 : center_kx + self.acs_block_size[0] // 2,
            center_ky - self.acs_block_size[1] // 2 : center_ky + self.acs_block_size[1] // 2,
        ] = 0

        _mask = tmp_mask.view(-1) if tmp_mask.is_contiguous() else tmp_mask.reshape(-1)

        num_valid = torch.sum(_mask)
        ind = torch.multinomial(_mask / num_valid, int(self.rho * num_valid), replacement=False)
        _mask[ind] = 0

        return _mask.view(mask.shape)

    @staticmethod
    def __find_center_ind__(data: torch.Tensor, dims: tuple = (1, 2, 3)) -> int:
        """
        Calculates the center of the k-space.

        Parameters
        ----------
        data : torch.Tensor
            Input data. The shape should be (nx, ny, nc).
        dims : tuple, optional
            Dimensions to calculate the norm. Default is ``(1, 2, 3)``.

        Returns
        -------
        center_ind : int
            The center of the k-space
        """
        for dim in dims:
            data = torch.linalg.norm(data, dim=dim, keepdims=True)
        return torch.argsort(data.squeeze())[-1:]

    def __apply_outer_kspace_unmask__(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Applies outer k-space (un)mask.

        Parameters
        ----------
        mask : torch.Tensor
            Input mask. The shape should be (nx, ny).

        Returns
        -------
        mask : torch.Tensor
            Output mask. The shape should be (nx, ny).
        """
        mask_out = int(mask.shape[1] * self.outer_kspace_fraction)
        mask[:, 0:mask_out] = torch.ones((mask.shape[0], mask_out))
        mask[:, mask.shape[1] - mask_out : mask.shape[1]] = torch.ones((mask.shape[0], mask_out))
        return mask

    @staticmethod
    def __exists__(fname: str, shape: Tuple) -> Union[np.ndarray, None]:
        """
        Checks if the sampling mask exists.

        Parameters
        ----------
        fname : str
            Filename to save the sampling mask.
        shape : tuple
            Shape of the sampling mask.

        Returns
        -------
        exists : bool
            True if the sampling mask exists.
        """
        if ".h5" in fname:
            fname = fname.replace(".h5", ".npy")
        else:
            fname = fname + ".npy"
        # set path to the tmp directory of the home directory
        path = os.path.join(os.path.expanduser("~"), "tmp", fname)
        if os.path.exists(path):
            masks = np.load(path)
            if (masks.shape[2], masks.shape[3]) == shape:
                return masks
        return None

    @staticmethod
    def __export__(mask: torch.Tensor, fname: str) -> None:
        """
        Exports the sampling mask to a numpy file.

        Parameters
        ----------
        mask : torch.Tensor
            Sampling mask. The shape should be (1, nx, ny, 1).
        fname : str
            Filename to save the sampling mask.
        """
        if ".h5" in fname:
            fname = fname.replace(".h5", ".npy")
        else:
            fname = fname + ".npy"
        # set path to the tmp directory of the home directory
        path = os.path.join(os.path.expanduser("~"), "tmp", fname)
        np.save(path, mask.cpu().numpy())


class N2R:
    """
    Generates Noise to Reconstruction (N2R) sampling masks, as presented in [1]_.

    References
    ----------
    [1] AD Desai, BM Ozturkler, CM Sandino, et al. Noise2Recon: Enabling Joint MRI Reconstruction and Denoising with
        Semi-Supervised and Self-Supervised Learning. ArXiv 2022. https://arxiv.org/abs/2110.00075

    Parameters
    ----------
    probability : float, optional
        Probability of sampling. Default is ``0.0``.
    std_devs : Tuple[float, float], optional
        Standard deviations of the Gaussian noise. Default is ``(0.0, 0.0)``.
    rhos: Tuple[float, float], optional
        Rho values for the Gaussian noise. Default is ``(0.0, 0.0)``.
    use_mask : bool, optional
        Whether to use the mask. Default is ``True``.

    Returns
    -------
    sampling_mask_noise : torch.Tensor
        Sampling mask with noise. The shape should be (1, nx, ny, 1).
    """

    def __init__(  # noqa: W0221
        self,
        probability: float = 0.0,
        std_devs: Tuple[float, float] = (0.0, 0.0),
        rhos: Tuple[float, float] = (0.0, 0.0),
        use_mask: bool = True,
    ):
        self.probability = probability
        self.std_devs = std_devs
        self.rhos = rhos
        self.use_mask = use_mask

    def __call__(self, data: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Generates N2R sampling masks.

        Parameters
        ----------
        data : torch.Tensor
            Input data. The shape should be (nc, nx, ny).
        mask : torch.Tensor
            Input mask. The shape should be (nx, ny).

        Returns
        -------
        sampling_mask_noise : torch.Tensor
            Sampling mask with noise. The shape should be (1, nx, ny, 1).
        """
        mask = mask.squeeze(0).squeeze(-1)
        # if mask is 1D, repeat it for nx
        if mask.shape[0] == 1:
            mask = mask.repeat_interleave(data.shape[1], 0)
        return self.forward(mask)

    def __repr__(self):
        return (
            f"N2R(probability={self.probability}, std_devs={self.std_devs}, rhos={self.rhos}, "
            f"use_mask={self.use_mask})"
        )

    def __str__(self):
        return self.__repr__()

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Generates N2R sampling masks.

        Parameters
        ----------
        mask : torch.Tensor
            Input mask. The shape should be (nx, ny).

        Returns
        -------
        sampling_mask_noise : torch.Tensor
            Sampling mask with noise. The shape should be (1, nx, ny, 1).
        """
        _rand = torch.rand(1).item()

        if _rand >= self.probability:
            return torch.ones_like(mask).unsqueeze(0).unsqueeze(-1)

        rhos = (
            self._rand_range(*self.rhos)
            if self.rhos is not None and self.rhos[0] != 0.0 and self.rhos[1] != 0.0
            else None
        )

        if not self.use_mask:
            mask = torch.ones(mask.shape)

        gen = torch.Generator(device=mask.device).manual_seed(int(_rand * 1e10))
        noise = torch.view_as_complex(
            self._rand_range(*self.std_devs) * torch.randn(mask.shape + (2,), generator=gen, device=mask.device),
        )

        if rhos is not None and rhos != 1:
            shape = mask.shape
            mask = mask.view(-1)
            # TODO: this doesn't work if the matrix is > 2*24 in size.
            num_valid = torch.sum(mask)
            weights = mask / num_valid
            samples = torch.multinomial(weights, int((1 - rhos) * num_valid), replacement=False, generator=gen)
            mask[samples] = 0
            mask = mask.view(shape)

        if mask is not None:
            noise = noise * mask

        return torch.abs(noise).to(mask).unsqueeze(0).unsqueeze(-1)

    @staticmethod
    def _rand_range(low, high, size: int = None) -> float:
        """
        Uniform float random number between [low, high).

        Parameters
        ----------
        low : float
            Lower bound.
        high : float
            Upper bound.
        size : int, optional
            Number of samples. Default is ``None``.

        Returns
        -------
        val : float
            A uniformly sampled number in range [low, high).
        """
        if size is None:
            size = 1
        if low > high:
            high, low = low, high
        if high - low == 0:
            return low
        return (low + (high - low) * torch.rand(size)).cpu().item()


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
    kspace_normalization: str, optional
        Normalize in k-space.
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
    >>> data = torch.randn(2, 2, 2, 2, 2)  1j * torch.randn(2, 2, 2, 2, 2)
    >>> print(torch.min(torch.abs(data)), torch.max(torch.abs(data)))
    tensor(1e-06) tensor(1.4142)
    >>> normalizer = Normalizer(normalization_type="max")
    >>> normalized_data = normalizer(data)
    >>> print(torch.min(torch.abs(data)), torch.max(torch.abs(data)))
    tensor(0.) tensor(1.)
    """

    def __init__(  # noqa: W0221
        self,
        normalization_type: Optional[str] = None,
        kspace_normalization: bool = False,
        fft_centered: bool = False,
        fft_normalization: str = "backward",
        spatial_dims: Sequence[int] = (-2, -1),
    ):
        self.normalization_type = normalization_type
        self.kspace_normalization = kspace_normalization
        self.fft_centered = fft_centered
        self.fft_normalization = fft_normalization
        self.spatial_dims = spatial_dims

    def __call__(
        self,
        data: Union[torch.Tensor, List[torch.Tensor], None],
        apply_backward_transform: bool = False,
        apply_forward_transform: bool = False,
    ) -> List[Union[List[torch.Tensor], torch.Tensor], Union[Dict, None]]:  # type: ignore
        if not utils.is_none(data):
            if isinstance(data, list) and len(data) > 0:
                return [self.forward(d, apply_backward_transform, apply_forward_transform) for d in data]
            if data.dim() > 1 and data.mean() != 1:  # type: ignore
                return self.forward(data, apply_backward_transform, apply_forward_transform)
        return data, None  # type: ignore

    def __repr__(self):
        return f"Normalization type is set to {self.normalization_type}."

    def __str__(self):
        return self.__repr__()

    def forward(
        self,
        data: torch.Tensor,
        apply_backward_transform: bool = False,
        apply_forward_transform: bool = False,
    ) -> List[torch.Tensor, Dict]:  # type: ignore
        if self.kspace_normalization and apply_backward_transform:
            apply_backward_transform = False

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

        attrs = {
            "max": torch.max(torch.abs(data)),
            "mean": torch.mean(torch.abs(data)),
            "std": torch.std(torch.abs(data)),
            "min": torch.min(torch.abs(data)),
        }

        if self.normalization_type == "max":
            data = data / torch.max(torch.abs(data))
        elif self.normalization_type == "mean":
            data = data - torch.mean(torch.abs(data))
            data = data / torch.std(torch.abs(data))
        elif self.normalization_type == "minmax":
            min_value = torch.min(torch.abs(data))
            data = (data - min_value) / (torch.max(torch.abs(data)) - min_value)
        elif utils.is_none(self.normalization_type):  # type: ignore
            pass

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

        return data, attrs  # type: ignore


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
    >>> data = torch.randn(2, 2, 2, 2, 2)  1j * torch.randn(2, 2, 2, 2, 2)
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
        Fraction of the outer k-space to be kept/unmasked. Default is ``0.0``.
    ssdu_export_and_reuse_masks: bool, optional
        Whether to export and reuse the masks. Default is ``False``.
    n2r : bool, optional
        Whether to apply Noise to Reconstruction (N2R) masks. Default is ``False``.
    n2r_supervised_rate : Optional[float], optional
        A float between 0 and 1. This controls what fraction of the subjects should be loaded for Noise to
        Reconstruction (N2R) supervised loss, if N2R is enabled. Default is ``0.0``.
    n2r_probability: float, optional
        Probability of applying N2R. Default is ``0.0``.
    n2r_std_devs: Tuple[float, float], optional
        Standard deviations for the noise. Default is ``(0.0, 0.0)``.
    n2r_rhos: Tuple[float, float], optional
        Rho values for the noise. Default is ``(0.0, 0.0)``.
    n2r_use_mask: bool, optional
        Whether to use a mask for N2R. Default is ``False``.
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
    MRIDataTransforms
        Preprocessed data.
    """

    def __init__(  # noqa: C901
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
        ssdu: bool = False,
        ssdu_mask_type: str = "Gaussian",
        ssdu_rho: float = 0.4,
        ssdu_acs_block_size: Sequence[int] = (4, 4),
        ssdu_gaussian_std_scaling_factor: float = 4.0,
        ssdu_outer_kspace_fraction: float = 0.0,
        ssdu_export_and_reuse_masks: bool = False,
        n2r: bool = False,
        n2r_supervised_rate: float = 0.0,
        n2r_probability: float = 0.0,
        n2r_std_devs: Tuple[float, float] = None,
        n2r_rhos: Tuple[float, float] = None,
        n2r_use_mask: bool = False,
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
        consecutive_slices: int = 1,  # noqa: B008
        use_seed: bool = True,
    ):
        super().__init__()

        self.coil_combination_method = coil_combination_method
        self.kspace_crop = kspace_crop
        self.crop_before_masking = crop_before_masking

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

        self.shift_mask = shift_mask
        self.masking = Masker(
            mask_func=mask_func,  # type: ignore
            spatial_dims=self.spatial_dims,  # type: ignore
            shift_mask=shift_mask,
            half_scan_percentage=half_scan_percentage,
            center_scale=mask_center_scale,  # type: ignore
            dimensionality=dimensionality,
            remask=remask,
        )

        self.n2r = n2r
        self.n2r_supervised_rate = n2r_supervised_rate
        self.n2r_masking = (
            N2R(
                probability=n2r_probability,
                std_devs=n2r_std_devs,  # type: ignore
                rhos=n2r_rhos,  # type: ignore
                use_mask=n2r_use_mask,
            )
            if self.n2r
            else None
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

        self.normalization = (
            Normalizer(
                normalization_type=normalization_type,
                kspace_normalization=kspace_normalization,
                fft_centered=self.fft_centered,
                fft_normalization=self.fft_normalization,
                spatial_dims=self.spatial_dims,  # type: ignore
            )
            if normalize_inputs
            else None
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
        self.cropping = Composer([self.cropping])  # type: ignore
        self.normalization = Composer([self.normalization])  # type: ignore

        self.use_seed = use_seed

    def __call__(  # noqa: C901
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
        Union[Union[List, torch.Tensor], torch.Tensor],
        Union[torch.Tensor, Any],
        str,
        int,
        Union[List, Any],
        Dict,
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
        kspace, masked_kspace, mask, kspace_pre_normalization_vars, acc = self.__process_kspace__(
            kspace, mask, attrs, fname
        )
        sensitivity_map, sensitivity_pre_normalization_vars = self.__process_coil_sensitivities_map__(
            sensitivity_map, kspace
        )
        if self.n2r and len(masked_kspace) > 1:
            prediction, prediction_pre_normalization_vars = self.__initialize_prediction__(
                prediction, masked_kspace[0], sensitivity_map
            )
            if isinstance(masked_kspace, list) and not masked_kspace[1][0].dim() < 2:
                noise_prediction, noise_prediction_pre_normalization_vars = self.__initialize_prediction__(
                    None, masked_kspace[1], sensitivity_map
                )
            else:
                noise_prediction = torch.tensor([])
                noise_prediction_pre_normalization_vars = None
            prediction = [prediction, noise_prediction]
        else:
            prediction, prediction_pre_normalization_vars = self.__initialize_prediction__(
                prediction, masked_kspace, sensitivity_map
            )
            noise_prediction_pre_normalization_vars = None
        target, target_pre_normalization_vars = self.__initialize_prediction__(
            None if self.ssdu else target, kspace, sensitivity_map
        )

        attrs.update(
            self.__parse_normalization_vars__(
                kspace_pre_normalization_vars,
                sensitivity_pre_normalization_vars,
                prediction_pre_normalization_vars,
                noise_prediction_pre_normalization_vars,
                target_pre_normalization_vars,
            )
        )

        return kspace, masked_kspace, sensitivity_map, mask, prediction, target, fname, slice_idx, acc, attrs

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
        Dict,
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
        List[
        Union[List[torch.Tensor], torch.Tensor],
        Union[List[torch.Tensor], torch.Tensor],
        Union[List[torch.Tensor], torch.Tensor],
        Dict,
        Union[List[Union[float, torch.Tensor, Any]]]
        ]
            The transformed kspace, the masked kspace, the mask, the attributes and the acceleration factor.
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

        init_kspace = kspace
        init_masked_kspace = masked_kspace
        init_mask = mask

        if "None" not in self.normalization.__repr__():
            if isinstance(kspace, list):
                kspaces = []
                pre_normalization_vars = []
                for i in range(len(kspace)):
                    _kspace, _pre_normalization_vars = self.normalization(  # type: ignore
                        kspace[i], apply_backward_transform=True
                    )
                    kspaces.append(_kspace)
                    pre_normalization_vars.append(_pre_normalization_vars)
                kspace = kspaces
            else:
                kspace, pre_normalization_vars = self.normalization(kspace, apply_backward_transform=True)  # type: ignore

            if isinstance(masked_kspace, list):
                masked_kspaces = []
                masked_pre_normalization_vars = []
                for i in range(len(masked_kspace)):
                    _masked_kspace, _masked_pre_normalization_vars = self.normalization(  # type: ignore
                        masked_kspace[i], apply_backward_transform=True
                    )
                    masked_kspaces.append(_masked_kspace)
                    masked_pre_normalization_vars.append(_masked_pre_normalization_vars)
                masked_kspace = masked_kspaces
            else:
                masked_kspace, masked_pre_normalization_vars = self.normalization(
                    masked_kspace, apply_backward_transform=True
                )
        else:
            pre_normalization_vars = None
            masked_pre_normalization_vars = None

        if self.ssdu:
            kspace, masked_kspace, mask = self.__self_supervised_data_undersampling__(  # type: ignore
                kspace, masked_kspace, mask, fname
            )

        n2r_pre_normalization_vars = None
        if self.n2r and (not attrs["n2r_supervised"] or self.ssdu):
            n2r_masked_kspace, n2r_mask = self.__noise_to_reconstruction__(init_kspace, init_masked_kspace, init_mask)

            if self.ssdu:
                if isinstance(mask, list):
                    for i in range(len(mask)):
                        if init_mask[i].dim() != mask[i][0].dim():  # type: ignore
                            # find dimensions == 1 in mask[i][0] and add them to init_mask
                            unitary_dims = [j for j in range(mask[i][0].dim()) if mask[i][0].shape[j] == 1]
                            # unsqueeze init_mask to the index of the unitary dimensions
                            for j in unitary_dims:
                                init_mask[i] = init_mask[i].unsqueeze(j)  # type: ignore
                        masked_kspace[i] = init_masked_kspace[i]
                        mask[i][0] = init_mask[i]
                else:
                    if init_mask.dim() != mask[0].dim():
                        # find dimensions == 1 in mask[0] and add them to init_mask
                        unitary_dims = [j for j in range(mask[0].dim()) if mask[0].shape[j] == 1]
                        # unsqueeze init_mask to the index of the unitary dimensions
                        for j in unitary_dims:
                            init_mask = init_mask.unsqueeze(j)
                    masked_kspace = init_masked_kspace
                    mask[0] = init_mask

            if "None" not in self.normalization.__repr__():
                if isinstance(masked_kspace, list):
                    masked_kspaces = []
                    masked_pre_normalization_vars = []
                    for i in range(len(masked_kspace)):
                        _masked_kspace, _masked_pre_normalization_vars = self.normalization(  # type: ignore
                            masked_kspace[i], apply_backward_transform=True
                        )
                        masked_kspaces.append(_masked_kspace)
                        masked_pre_normalization_vars.append(_masked_pre_normalization_vars)
                    masked_kspace = masked_kspaces
                else:
                    masked_kspace, masked_pre_normalization_vars = self.normalization(  # type: ignore
                        masked_kspace, apply_backward_transform=True
                    )
                if isinstance(n2r_masked_kspace, list):
                    n2r_masked_kspaces = []
                    n2r_pre_normalization_vars = []
                    for i in range(len(n2r_masked_kspace)):
                        _n2r_masked_kspace, _n2r_pre_normalization_vars = self.normalization(  # type: ignore
                            n2r_masked_kspace[i], apply_backward_transform=True
                        )
                        n2r_masked_kspaces.append(_n2r_masked_kspace)
                        n2r_pre_normalization_vars.append(_n2r_pre_normalization_vars)
                    n2r_masked_kspace = n2r_masked_kspaces
                else:
                    n2r_masked_kspace, n2r_pre_normalization_vars = self.normalization(  # type: ignore
                        n2r_masked_kspace, apply_backward_transform=True
                    )
            else:
                masked_pre_normalization_vars = None
                n2r_pre_normalization_vars = None  # type: ignore

            masked_kspace = [masked_kspace, n2r_masked_kspace]
            mask = [mask, n2r_mask]

        pre_normalization_vars = {  # type: ignore
            "kspace_pre_normalization_vars": pre_normalization_vars,
            "masked_kspace_pre_normalization_vars": masked_pre_normalization_vars,
            "noise_masked_kspace_pre_normalization_vars": n2r_pre_normalization_vars,
        }

        return kspace, masked_kspace, mask, pre_normalization_vars, acc  # type: ignore

    def __noise_to_reconstruction__(
        self,
        kspace: torch.Tensor,
        masked_kspace: torch.Tensor,
        mask: Union[List, torch.Tensor],
    ) -> Tuple[Union[List, torch.Tensor], Union[List, torch.Tensor]]:
        """
        Apply the noise-to-reconstruction transform.

        Parameters
        ----------
        kspace : torch.Tensor
            The fully-sampled kspace.
        masked_kspace : torch.Tensor
            The undersampled kspace.
        mask : Union[List, torch.Tensor]
            The undersampling mask.

        Returns
        -------
        n2r_masked_kspace : Union[List, torch.Tensor]
            The noise-to-reconstruction undersampled kspace.
        n2r_mask : Union[List, torch.Tensor]
            The noise-to-reconstruction mask.
        """
        if isinstance(mask, list):
            n2r_masked_kspaces = []
            n2r_masks = []
            for i in range(len(mask)):
                n2r_mask = self.n2r_masking(kspace, mask[i])  # type: ignore
                n2r_masks.append(n2r_mask)
                n2r_masked_kspaces.append(masked_kspace[i] * n2r_mask + 0.0)
            n2r_mask = n2r_masks
            n2r_masked_kspace = n2r_masked_kspaces
        else:
            n2r_mask = self.n2r_masking(kspace, mask)  # type: ignore
            n2r_masked_kspace = masked_kspace * n2r_mask + 0.0
        return n2r_masked_kspace, n2r_mask

    def __self_supervised_data_undersampling__(
        self,
        kspace: torch.Tensor,
        masked_kspace: Union[List, torch.Tensor],
        mask: Union[List, torch.Tensor],
        fname: str,
    ) -> Tuple[
        List[float | Any] | float | Any,
        List[float | Any] | float | Any,
        List[List[torch.Tensor | Any]] | List[torch.Tensor | Any],
    ]:
        """
        Self-supervised data undersampling.

        Parameters
        ----------
        kspace : torch.Tensor
            The fully-sampled kspace.
        masked_kspace : Union[List, torch.Tensor]
            The undersampled kspace.
        mask : Union[List, torch.Tensor]
            The undersampling mask.
        fname : str
            The filename of the current sample.

        Returns
        -------
        kspace : torch.Tensor
            The kspace with the loss mask applied.
        masked_kspace : torch.Tensor
            The kspace with the train mask applied.
        mask : list, [torch.Tensor, torch.Tensor]
            The train and loss masks.
        """
        if isinstance(mask, list):
            kspaces = []
            masked_kspaces = []
            masks = []
            for i in range(len(mask)):  # noqa: C0200
                is_1d = mask[i].squeeze().dim() == 1
                if self.shift_mask:
                    mask[i] = torch.fft.fftshift(mask[i].squeeze(-1), dim=(-2, -1)).unsqueeze(-1)
                mask[i] = mask[i].squeeze()
                if is_1d:
                    mask[i] = mask[i].unsqueeze(0).repeat_interleave(kspace.shape[1], dim=0)
                train_mask, loss_mask = self.ssdu_masking(kspace, mask[i], fname)  # type: ignore  # noqa: E1102
                if self.shift_mask:
                    train_mask = torch.fft.fftshift(train_mask, dim=(0, 1))
                    loss_mask = torch.fft.fftshift(loss_mask, dim=(0, 1))
                if is_1d:
                    train_mask = train_mask.unsqueeze(0).unsqueeze(-1)
                    loss_mask = loss_mask.unsqueeze(0).unsqueeze(-1)
                else:
                    # find unitary dims in mask
                    dims = [i for i, x in enumerate(mask[i].shape) if x == 1]
                    # unsqueeze to broadcast
                    for d in dims:
                        train_mask = train_mask.unsqueeze(d)
                        loss_mask = loss_mask.unsqueeze(d)
                if train_mask.dim() != kspace.dim():
                    # find dims != to any train_mask dim
                    dims = [i for i, x in enumerate(kspace.shape) if x not in train_mask.shape]
                    # unsqueeze to broadcast
                    for d in dims:
                        train_mask = train_mask.unsqueeze(d)
                        loss_mask = loss_mask.unsqueeze(d)
                kspaces.append(kspace * loss_mask + 0.0)
                masked_kspaces.append(masked_kspace[i] * train_mask + 0.0)  # type: ignore
                masks.append([train_mask, loss_mask])
            kspace = kspaces
            masked_kspace = masked_kspaces
            mask = masks
        else:
            is_1d = mask.squeeze().dim() == 1  # type: ignore
            if self.shift_mask:
                mask = torch.fft.fftshift(mask.squeeze(-1), dim=(-2, -1)).unsqueeze(-1)  # type: ignore
            mask = mask.squeeze()
            if is_1d:
                mask = mask.unsqueeze(0).repeat_interleave(kspace.shape[1], dim=0)
            train_mask, loss_mask = self.ssdu_masking(kspace, mask, fname)  # type: ignore  # noqa: E1102
            if self.shift_mask:
                train_mask = torch.fft.fftshift(train_mask, dim=(0, 1))
                loss_mask = torch.fft.fftshift(loss_mask, dim=(0, 1))
            if is_1d:
                train_mask = train_mask.unsqueeze(0).unsqueeze(-1)
                loss_mask = loss_mask.unsqueeze(0).unsqueeze(-1)
            else:
                # find unitary dims in mask
                dims = [i for i, x in enumerate(mask.shape) if x == 1]
                # unsqueeze to broadcast
                for d in dims:
                    train_mask = train_mask.unsqueeze(d)
                    loss_mask = loss_mask.unsqueeze(d)
            if train_mask.dim() != kspace.dim():
                # find dims != to any train_mask dim
                dims = [i for i, x in enumerate(kspace.shape) if x not in train_mask.shape]
                # unsqueeze to broadcast
                for d in dims:
                    train_mask = train_mask.unsqueeze(d)
                    loss_mask = loss_mask.unsqueeze(d)
            kspace = kspace * loss_mask + 0.0
            masked_kspace = masked_kspace * train_mask + 0.0
            mask = [train_mask, loss_mask]
        return kspace, masked_kspace, mask

    def __process_coil_sensitivities_map__(
        self, sensitivity_map: np.ndarray, kspace: torch.Tensor
    ) -> Union[torch.Tensor, Dict]:
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
        List[torch.Tensor, Dict]
            The preprocessed coil sensitivities map and the normalization variables.
        """
        # This condition is necessary in case of auto estimation of sense maps.
        if sensitivity_map is not None and sensitivity_map.size != 0:
            sensitivity_map = utils.to_tensor(sensitivity_map)
        else:
            # If no sensitivity map is provided, either the data is singlecoil or the sense net is used.
            # Initialize the sensitivity map to 1 to assure for the singlecoil case.
            sensitivity_map = torch.ones_like(kspace)
        sensitivity_map = self.coils_shape_transforms(sensitivity_map, apply_forward_transform=True)
        sensitivity_map = self.cropping(sensitivity_map, apply_forward_transform=self.kspace_crop)  # type: ignore
        if "None" not in self.normalization.__repr__():
            sensitivity_map, pre_normalization_vars = self.normalization(  # type: ignore
                sensitivity_map, apply_forward_transform=self.kspace_crop
            )
        else:
            pre_normalization_vars = None
        return sensitivity_map, pre_normalization_vars

    def __initialize_prediction__(
        self, prediction: Union[np.ndarray, None], kspace: torch.Tensor, sensitivity_map: torch.Tensor
    ) -> List[Union[List[torch.Tensor], torch.Tensor], Dict]:  # type: ignore
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
        List[Union[List[torch.Tensor], torch.Tensor], Dict]
            The initialized prediction, either a list of coil-combined images or a single coil-combined image and the
            pre-normalization variables (min, max, mean, std).
        """
        if utils.is_none(prediction) or prediction.ndim < 2 or isinstance(kspace, list):  # type: ignore
            if isinstance(kspace, list):
                prediction = []
                pre_normalization_vars = []
                for y in kspace:
                    pred = self.init_reconstructor(
                        y, sensitivity_map, torch.empty([]), torch.empty([]), torch.empty([])
                    )
                    pred = self.cropping(pred, apply_forward_transform=self.kspace_crop)  # type: ignore
                    if "None" not in self.normalization.__repr__():
                        pred, _pre_normalization_vars = self.normalization(  # type: ignore
                            pred, apply_forward_transform=self.kspace_crop
                        )
                    else:
                        _pre_normalization_vars = None
                    prediction.append(pred)
                    pre_normalization_vars.append(_pre_normalization_vars)
                if prediction[0].shape[-1] != 2:
                    prediction = [torch.view_as_real(x) for x in prediction]
            else:
                prediction = self.init_reconstructor(
                    kspace, sensitivity_map, torch.empty([]), torch.empty([]), torch.empty([])
                )
                prediction = self.cropping(prediction, apply_forward_transform=self.kspace_crop)  # type: ignore
                if "None" not in self.normalization.__repr__():
                    prediction, pre_normalization_vars = self.normalization(  # type: ignore
                        prediction, apply_forward_transform=self.kspace_crop
                    )
                else:
                    pre_normalization_vars = None
                if prediction.shape[-1] != 2:  # type: ignore
                    prediction = torch.view_as_real(prediction)
        else:
            if isinstance(prediction, np.ndarray):
                prediction = utils.to_tensor(prediction)
            prediction = self.cropping(prediction, apply_forward_transform=self.kspace_crop)  # type: ignore
            if "None" not in self.normalization.__repr__():
                prediction, pre_normalization_vars = self.normalization(  # type: ignore
                    prediction, apply_forward_transform=self.kspace_crop
                )
            else:
                pre_normalization_vars = None
            if prediction.shape[-1] != 2 and prediction.type == "torch.ComplexTensor":  # type: ignore
                prediction = torch.view_as_real(prediction)
        return prediction, pre_normalization_vars  # type: ignore

    def __parse_normalization_vars__(
        self, kspace_vars, sensitivity_vars, prediction_vars, noise_prediction_vars, target_vars
    ) -> Dict:
        """
        Parses the normalization variables and returns a unified dictionary.

        Parameters
        ----------
        kspace_vars : Dict
            The kspace normalization variables.
        sensitivity_vars : Dict
            The sensitivity map normalization variables.
        prediction_vars : Dict
            The prediction normalization variables.
        noise_prediction_vars : Union[Dict, None]
            The noise prediction normalization variables.
        target_vars : Dict
            The target normalization variables.

        Returns
        -------
        Dict
            The normalization variables.
        """
        normalization_vars = {}
        masked_kspace_vars = kspace_vars["masked_kspace_pre_normalization_vars"]
        if isinstance(masked_kspace_vars, list):
            if masked_kspace_vars[0] is not None:
                for i, masked_kspace_var in enumerate(masked_kspace_vars):
                    normalization_vars[f"masked_kspace_min_{i}"] = masked_kspace_var["min"]
                    normalization_vars[f"masked_kspace_max_{i}"] = masked_kspace_var["max"]
                    normalization_vars[f"masked_kspace_mean_{i}"] = masked_kspace_var["mean"]
                    normalization_vars[f"masked_kspace_std_{i}"] = masked_kspace_var["std"]
        else:
            if masked_kspace_vars is not None:
                normalization_vars["masked_kspace_min"] = masked_kspace_vars["min"]
                normalization_vars["masked_kspace_max"] = masked_kspace_vars["max"]
                normalization_vars["masked_kspace_mean"] = masked_kspace_vars["mean"]
                normalization_vars["masked_kspace_std"] = masked_kspace_vars["std"]
        noise_masked_kspace_vars = kspace_vars["noise_masked_kspace_pre_normalization_vars"]
        if noise_masked_kspace_vars is not None:
            if isinstance(noise_masked_kspace_vars, list):
                if noise_masked_kspace_vars[0] is not None:
                    for i, noise_masked_kspace_var in enumerate(noise_masked_kspace_vars):
                        normalization_vars[f"noise_masked_kspace_min_{i}"] = noise_masked_kspace_var["min"]
                        normalization_vars[f"noise_masked_kspace_max_{i}"] = noise_masked_kspace_var["max"]
                        normalization_vars[f"noise_masked_kspace_mean_{i}"] = noise_masked_kspace_var["mean"]
                        normalization_vars[f"noise_masked_kspace_std_{i}"] = noise_masked_kspace_var["std"]
            else:
                if noise_masked_kspace_vars is not None:
                    normalization_vars["noise_masked_kspace_min"] = noise_masked_kspace_vars["min"]
                    normalization_vars["noise_masked_kspace_max"] = noise_masked_kspace_vars["max"]
                    normalization_vars["noise_masked_kspace_mean"] = noise_masked_kspace_vars["mean"]
                    normalization_vars["noise_masked_kspace_std"] = noise_masked_kspace_vars["std"]
        kspace_vars = kspace_vars["kspace_pre_normalization_vars"]
        if isinstance(kspace_vars, list):
            if kspace_vars[0] is not None:
                for i, kspace_var in enumerate(kspace_vars):
                    normalization_vars[f"kspace_min_{i}"] = kspace_var["min"]
                    normalization_vars[f"kspace_max_{i}"] = kspace_var["max"]
                    normalization_vars[f"kspace_mean_{i}"] = kspace_var["mean"]
                    normalization_vars[f"kspace_std_{i}"] = kspace_var["std"]
        else:
            if kspace_vars is not None:
                normalization_vars["kspace_min"] = kspace_vars["min"]
                normalization_vars["kspace_max"] = kspace_vars["max"]
                normalization_vars["kspace_mean"] = kspace_vars["mean"]
                normalization_vars["kspace_std"] = kspace_vars["std"]
        if sensitivity_vars is not None:
            normalization_vars["sensitivity_maps_min"] = sensitivity_vars["min"]
            normalization_vars["sensitivity_maps_max"] = sensitivity_vars["max"]
            normalization_vars["sensitivity_maps_mean"] = sensitivity_vars["mean"]
            normalization_vars["sensitivity_maps_std"] = sensitivity_vars["std"]
        if isinstance(prediction_vars, list):
            if prediction_vars[0] is not None:
                for i, prediction_var in enumerate(prediction_vars):
                    normalization_vars[f"prediction_min_{i}"] = prediction_var["min"]
                    normalization_vars[f"prediction_max_{i}"] = prediction_var["max"]
                    normalization_vars[f"prediction_mean_{i}"] = prediction_var["mean"]
                    normalization_vars[f"prediction_std_{i}"] = prediction_var["std"]
        else:
            if prediction_vars is not None:
                normalization_vars["prediction_min"] = prediction_vars["min"]
                normalization_vars["prediction_max"] = prediction_vars["max"]
                normalization_vars["prediction_mean"] = prediction_vars["mean"]
                normalization_vars["prediction_std"] = prediction_vars["std"]
        if noise_prediction_vars is not None:
            if isinstance(noise_prediction_vars, list):
                for i, noise_prediction_var in enumerate(noise_prediction_vars):
                    normalization_vars[f"noise_prediction_min_{i}"] = noise_prediction_var["min"]
                    normalization_vars[f"noise_prediction_max_{i}"] = noise_prediction_var["max"]
                    normalization_vars[f"noise_prediction_mean_{i}"] = noise_prediction_var["mean"]
                    normalization_vars[f"noise_prediction_std_{i}"] = noise_prediction_var["std"]
            else:
                normalization_vars["noise_prediction_min"] = noise_prediction_vars["min"]
                normalization_vars["noise_prediction_max"] = noise_prediction_vars["max"]
                normalization_vars["noise_prediction_mean"] = noise_prediction_vars["mean"]
                normalization_vars["noise_prediction_std"] = noise_prediction_vars["std"]
        if isinstance(target_vars, list):
            if target_vars[0] is not None:
                for i, target_var in enumerate(target_vars):
                    normalization_vars[f"target_min_{i}"] = target_var["min"]
                    normalization_vars[f"target_max_{i}"] = target_var["max"]
                    normalization_vars[f"target_mean_{i}"] = target_var["mean"]
                    normalization_vars[f"target_std_{i}"] = target_var["std"]
        else:
            if target_vars is not None:
                normalization_vars["target_min"] = target_vars["min"]
                normalization_vars["target_max"] = target_vars["max"]
                normalization_vars["target_mean"] = target_vars["mean"]
                normalization_vars["target_std"] = target_vars["std"]
        return normalization_vars
