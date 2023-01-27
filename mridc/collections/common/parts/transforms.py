# coding=utf-8
from __future__ import annotations

from math import sqrt
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from mridc.collections.common.parts import fft as fft
from mridc.collections.common.parts import utils as utils


class NoisePreWhitening:
    """
    Applies noise pre-whitening / coil decorrelation.

    Parameters
    ----------
    patch_size : list of ints
        Define patch size to calculate psi, [x_start, x_end, y_start, y_end].
    scale_factor : float
        Applied on the noise covariance matrix. Used to adjust for effective noise bandwidth and difference in
        sampling rate between noise calibration and actual measurement.
        scale_factor = (T_acq_dwell/T_noise_dwell)*NoiseReceiverBandwidthRatio

    Examples
    --------
    >>> import torch
    >>> from mridc.collections.common.parts.transforms import NoisePreWhitening
    >>> data = torch.randn([30, 100, 100], dtype=torch.complex64)
    >>> data = torch.view_as_real(data)
    >>> data.mean()
    tensor(-0.0011)
    >>> noise_prewhitening = NoisePreWhitening(patch_size=[10, 40, 10, 40], scale_factor=1.0)
    >>> noise_prewhitening(data).mean()
    tensor(-0.0023)
    """

    def __init__(self, patch_size: List[int], scale_factor: float = 1.0):
        super().__init__()
        self.patch_size = patch_size
        self.scale_factor = scale_factor

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return self.forward(data)

    def __repr__(self):
        return f"Noise pre-whitening is applied with patch size {self.patch_size}."

    def __str__(self):
        return str(self.__repr__)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        data : torch.Tensor
            Input data to apply noise pre-whitening.

        Returns
        -------
        torch.Tensor
            Noise pre-whitened data.
        """
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
        spatial_dims: Sequence[int] = None,
    ):
        super().__init__()
        self.virtual_coils = virtual_coils
        self.calib_lines = calib_lines
        self.align_data = align_data
        self.fft_centered = fft_centered
        self.fft_normalization = fft_normalization
        self.spatial_dims = spatial_dims

    def __call__(self, data):
        return self.forward(data)

    def __repr__(self):
        return f"Coil Compression is applied reducing coils from {self.coils} to {self.virtual_coils}."

    def __str__(self):
        return str(self.__repr__)

    def forward(self, data):
        """
        Parameters
        ----------
        data : torch.Tensor
            Input data to apply coil compression.

        Returns
        -------
        torch.Tensor
            Coil compressed data.
        """
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

        return fft.fft2(
            torch.view_as_real(rotated_compressed_data.permute(2, 0, 1)),
            self.fft_centered,
            self.fft_normalization,
            self.spatial_dims,
        )

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

    def __init__(self, zero_filling_size: Tuple, spatial_dims: Tuple = (-2, -1)):
        self.zero_filling_size = zero_filling_size
        self.spatial_dims = spatial_dims

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Applies zero filling to data."""
        return self.forward(data)

    def __repr__(self) -> str:
        return f"Zero-Filling will be applied to data with size {self.zero_filling_size}."

    def __str__(self) -> str:
        return self.__repr__()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Applies zero filling to data."""
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
        spatial_dims: Tuple = (-2, -1),
    ):
        self.cropping_size = cropping_size
        self.fft_centered = fft_centered
        self.fft_normalization = fft_normalization
        self.spatial_dims = spatial_dims

    def __call__(self, data: torch.Tensor, kspace_crop: bool = False) -> torch.Tensor:
        return self.forward(data, kspace_crop)

    def __repr__(self):
        return f"Data will be cropped to size={self.cropping_size}."

    def __str__(self):
        return self.__repr__()

    def forward(self, data: torch.Tensor, kspace_crop: bool = False) -> torch.Tensor:
        """Applies cropping to data."""
        if kspace_crop:
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

        if kspace_crop:
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
    padding : tuple, optional
        Padding size. Default is `None`.
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
    >>> masker = Masker(mask_func=None, spatial_dims=(-2, -1), padding=None, shift_mask=False, \
     half_scan_percentage=0.0, center_scale=0.02, dimensionality=2, remask=True)
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
        spatial_dims: Tuple = (-2, -1),
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
    ) -> Tuple[
        List[float | torch.Tensor | Any],
        List[torch.Tensor | Any] | List[torch.Tensor | np.ndarray | None | Any],
        List[int | torch.Tensor | Any],
    ]:
        """Applies mask to data."""
        if not utils.is_none(mask) and isinstance(mask, list) and len(mask) > 0:
            self.__type__ = "Masks are precomputed and loaded."
        elif not utils.is_none(mask) and not isinstance(mask, list) and mask.ndim != 0 and len(mask) > 0:  # type: ignore
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

        elif not utils.is_none(mask) and not isinstance(mask, list) and mask.ndim != 0 and len(mask) > 0:  # type: ignore
            if list(mask.shape) == [data.shape[self.spatial_dims[0]], data.shape[self.spatial_dims[1]]]:  # type: ignore
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
        spatial_dims: Tuple = (-2, -1),
    ):
        self.normalization_type = normalization_type
        self.fft_centered = fft_centered
        self.fft_normalization = fft_normalization
        self.spatial_dims = spatial_dims

    def __call__(self, data: torch.Tensor, apply_backward_transform: bool = False) -> List[Tensor] | Tensor:
        if isinstance(data, list):
            return [self.forward(d, apply_backward_transform) for d in data]
        return self.forward(data, apply_backward_transform)

    def __repr__(self):
        return f"Normalization type is set to {self.normalization_type}."

    def __str__(self):
        return self.__repr__()

    def forward(self, data: torch.Tensor, apply_backward_transform: bool = False) -> torch.Tensor:
        if apply_backward_transform:
            data = fft.ifft2(data, self.fft_centered, self.fft_normalization, self.spatial_dims)

        if self.normalization_type == "max":
            data = data / torch.max(torch.abs(data))
        elif self.normalization_type == "mean":
            data = data - torch.mean(torch.abs(data))
            data = data / torch.std(torch.abs(data))
        elif self.normalization_type == "minmax":
            min_value = torch.min(torch.abs(data))
            data = (data - min_value) / (torch.max(torch.abs(data)) - min_value)

        if apply_backward_transform:
            data = fft.fft2(data, self.fft_centered, self.fft_normalization, self.spatial_dims)

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

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            data = transform(data)
        return data

    def __repr__(self):
        return f"Composed transforms: {self.transforms}"

    def __str__(self):
        return self.__repr__()
