# coding=utf-8
__author__ = "Dimitrios Karkalousos"

# Parts of the code have been taken from https://github.com/facebookresearch/fastMRI

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import torch

from mridc.collections.common.data.subsample import MaskFunc

__all__ = [
    "is_none",
    "to_tensor",
    "complex_mul",
    "complex_conj",
    "complex_abs",
    "complex_abs_sq",
    "rss",
    "rss_complex",
    "sense",
    "coil_combination_method",
    "save_predictions",
    "check_stacked_complex",
    "apply_mask",
    "mask_center",
    "batched_mask_center",
    "center_crop",
    "complex_center_crop",
    "center_crop_to_smallest",
    "rnn_weights_init",
]


def is_none(x: Union[Any, None]) -> bool:
    """
    Check if input is None or "None".

    Parameters
    ----------
    x : Union[Any, None]
        Input to check.

    Returns
    -------
    bool
        True if x is None or "None", False otherwise.

    Examples
    --------
    >>> from mridc.collections.common.parts.utils import is_none
    >>> is_none(None)
    True
    >>> is_none("None")
    True
    """
    return x is None or str(x).lower() == "none"


def to_tensor(x: np.ndarray) -> torch.Tensor:
    """
    Converts a numpy array to a torch tensor. For complex arrays, the real and imaginary parts are stacked along the
    last dimension.

    Parameters
    ----------
    x : np.ndarray
        Input numpy array to be converted to torch.

    Returns
    -------
    torch.Tensor
        Torch tensor version of input.

    Examples
    --------
    >>> from mridc.collections.common.parts.utils import to_tensor
    >>> import numpy as np
    >>> data = np.array([[1+1j, 2+2j, 3+3j], [4+4j, 5+5j, 6+6j]])
    >>> data.shape
    (2, 3)
    >>> to_tensor(data)
    tensor([[[1., 1.],
            [2., 2.],
            [3., 3.]],
            [[4., 4.],
            [5., 5.],
            [6., 6.]]], dtype=torch.float64)
    >>> to_tensor(data).shape
    torch.Size([2, 3, 2])
    """
    if np.iscomplexobj(x):
        x = np.stack((x.real, x.imag), axis=-1)
    return torch.from_numpy(x)


def reshape_fortran(x, shape) -> torch.Tensor:
    """
    Reshapes a tensor in Fortran order. Taken from https://stackoverflow.com/a/63964246

    Parameters
    ----------
    x : torch.Tensor
        Input tensor to be reshaped.
    shape : Sequence[int]
        Shape to reshape the tensor to.

    Returns
    -------
    torch.Tensor
        Reshaped tensor.

    Examples
    --------
    >>> from mridc.collections.common.parts.utils import reshape_fortran
    >>> import torch
    >>> data = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    >>> data.shape
    torch.Size([2, 2, 3])
    >>> reshape_fortran(data, (3, 2, 2))
    tensor([[[ 1,  7],
            [ 4, 10]],
            [[ 2,  8],
            [ 5, 11]],
            [[ 3,  9],
            [ 6, 12]]])
    >>> reshape_fortran(data, (3, 2, 2)).shape
    torch.Size([3, 2, 2])
    """
    return x.permute(*reversed(range(len(x.shape)))).reshape(*reversed(shape)).permute(*reversed(range(len(shape))))


def complex_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Complex multiplication.

    This multiplies two complex tensors assuming that they are both stored as real arrays with the last dimension
    being the complex dimension.

    Parameters
    ----------
    x : torch.Tensor
        First complex tensor to multiply. The last dimension must be of size 2.
    y : torch.Tensor
        Second complex tensor to multiply. The last dimension must be of size 2.

    Returns
    -------
    torch.Tensor
        Result of complex multiplication.

    Examples
    --------
    >>> from mridc.collections.common.parts.utils import complex_mul
    >>> import torch
    >>> datax = torch.tensor([1+1j, 2+2j, 3+3j])
    >>> datay = torch.tensor([4+4j, 5+5j, 6+6j])
    >>> complex_mul(datax, datay)
    tensor([[-7.+20.j],
            [-4.+16.j],
            [-1.+12.j]])
    """
    if not x.shape[-1] == y.shape[-1] == 2:
        raise ValueError("Tensors do not have separate complex dim.")
    re = x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1]
    im = x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0]
    return torch.stack((re, im), dim=-1)


def complex_conj(x: torch.Tensor) -> torch.Tensor:
    """
    Complex conjugate.

    This applies the complex conjugate assuming that the input array has the last dimension as the complex dimension.

    Parameters
    ----------
    x : torch.Tensor
        Complex tensor to apply the complex conjugate to. The last dimension must be of size 2.

    Returns
    -------
    torch.Tensor
        Result of complex conjugate.

    Examples
    --------
    >>> from mridc.collections.common.parts.utils import complex_conj
    >>> import torch
    >>> data = torch.tensor([1+1j, 2+2j, 3+3j])
    >>> complex_conj(data)
    tensor([1.-1.j, 2.-2.j, 3.-3.j])
    """
    if x.shape[-1] != 2:
        raise ValueError("Tensor does not have separate complex dim.")
    return torch.stack((x[..., 0], -x[..., 1]), dim=-1)


def complex_abs(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the absolute value of a complex valued input tensor.

    Parameters
    ----------
    x : torch.Tensor
        Complex tensor. The last dimension must be of size 2.

    Returns
    -------
    torch.Tensor
        Absolute value of complex tensor.

    Examples
    --------
    >>> from mridc.collections.common.parts.utils import complex_abs
    >>> import torch
    >>> data = torch.tensor([1+1j, 2+2j, 3+3j])
    >>> complex_abs(data)
    tensor([1.4142, 2.8284, 4.2426])
    """
    if x.shape[-1] != 2:
        raise ValueError("Tensor does not have separate complex dim.")
    return (x**2).sum(dim=-1).sqrt()


def complex_abs_sq(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the squared absolute value of a complex tensor.

    Parameters
    ----------
    x : torch.Tensor
        Complex tensor. The last dimension must be of size 2.

    Returns
    -------
    torch.Tensor
        Squared absolute value of complex tensor.

    Examples
    --------
    >>> from mridc.collections.common.parts.utils import complex_abs_sq
    >>> import torch
    >>> data = torch.tensor([1+1j, 2+2j, 3+3j])
    >>> complex_abs_sq(data)
    tensor([2., 8., 18.])
    """
    if x.shape[-1] != 2:
        raise ValueError("Tensor does not have separate complex dim.")
    return (x**2).sum(dim=-1)


def check_stacked_complex(x: torch.Tensor) -> torch.Tensor:
    """
    Check if tensor is stacked complex (real & imaginary parts stacked along last dim) and convert it to a combined
    complex tensor.

    Parameters
    ----------
    x : torch.Tensor
        Tensor to check.

    Returns
    -------
    torch.Tensor
        Tensor with stacked complex converted to combined complex.

    Examples
    --------
    >>> from mridc.collections.common.parts.utils import check_stacked_complex
    >>> import torch
    >>> data = torch.tensor([1+1j, 2+2j, 3+3j])
    >>> data.shape
    torch.Size([3])
    >>> data = torch.view_as_real(data)
    >>> data.shape
    >>> check_stacked_complex(data)
    tensor([1.+1.j, 2.+2.j, 3.+3.j])
    >>> check_stacked_complex(data).shape
    torch.Size([3])
    >>> data = torch.tensor([1+1j, 2+2j, 3+3j])
    >>> data.shape
    torch.Size([3])
    >>> check_stacked_complex(data)
    tensor([1.+1.j, 2.+2.j, 3.+3.j])
    >>> check_stacked_complex(data).shape
    torch.Size([3])
    """
    return torch.view_as_complex(x) if x.shape[-1] == 2 else x


def rss(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Compute the Root Sum of Squares (RSS).

    RSS is computed assuming that dim is the coil dimension.

    Parameters
    ----------
    x : torch.Tensor
        Tensor to apply the RSS transform to.
    dim : int, optional
        Dimension to apply the RSS transform to. Default is ``0``.

    Returns
    -------
    torch.Tensor
        Coil-combined tensor with RSS applied.

    Examples
    --------
    >>> from mridc.collections.common.parts.utils import rss
    >>> import torch
    >>> data = torch.tensor([[[[1., 1.], [2., 2.], [3., 3.]], [[1., 1.], [2., 2.], [3., 3.]]], \
    [[[1., 1.], [2., 2.], [3., 3.]], [[1., 1.], [2., 2.], [3., 3.]]]])
    >>> data.shape
    torch.Size([2, 2, 3, 2])
    >>> rss(data)
    tensor([[[2.8284, 2.8284],
        [5.6569, 5.6569],
        [8.4853, 8.4853]],
        [[2.8284, 2.8284],
        [5.6569, 5.6569],
        [8.4853, 8.4853]]])
    >>> rss(data).shape
    torch.Size([2, 3, 2])
    """
    return torch.sqrt((x**2).sum(dim))


def rss_complex(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Compute the Root Sum of Squares (RSS) for complex inputs.

    RSS is computed assuming that dim is the coil dimension.

    Parameters
    ----------
    x : torch.Tensor
        Tensor to apply the RSS transform to.
    dim : int, optional
        Dimension to apply the RSS transform to. Default is ``0``.

    Returns
    -------
    torch.Tensor
        Coil-combined tensor with RSS applied.

    Examples
    --------
    >>> from mridc.collections.common.parts.utils import rss_complex
    >>> import torch
    >>> data = torch.tensor([[[1+1j, 2+2j, 3+3j], [1+1j, 2+2j, 3+3j]], [[1+1j, 2+2j, 3+3j], [1+1j, 2+2j, 3+3j]]])
    >>> data.shape
    torch.Size([2, 2, 3])
    >>> rss_complex(data, dim=0)
    tensor([[1.4142, 2.8284, 4.2426],
            [1.4142, 2.8284, 4.2426]])
    >>> rss_complex(data, dim=0).shape
    torch.Size([2, 3])
    """
    return torch.sqrt(complex_abs_sq(x).sum(dim))


def sense(x: torch.Tensor, sensitivity_maps: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Coil-combination according to the SENSitivity Encoding (SENSE) method [1].

    References
    ----------
    .. [1] Pruessmann KP, Weiger M, Scheidegger MB, Boesiger P. SENSE: Sensitivity encoding for fast MRI.
        Magn Reson Med 1999; 42:952-962.

    Parameters
    ----------
    x : torch.Tensor
        The tensor to coil-combine.
    sensitivity_maps : torch.Tensor
        The coil sensitivity maps.
    dim : int, optional
        The dimension to coil-combine along. Default is ``0``.

    Returns
    -------
    torch.Tensor
        Coil-combined tensor with SENSE applied.

    Examples
    --------
    >>> from mridc.collections.common.parts.utils import sense
    >>> import torch
    >>> data = torch.tensor([[[[1., 1.], [2., 2.], [3., 3.]], [[1., 1.], [2., 2.], [3., 3.]]], \
    [[[1., 1.], [2., 2.], [3., 3.]], [[1., 1.], [2., 2.], [3., 3.]]]])
    >>> data.shape
    torch.Size([2, 2, 3, 2])
    >>> coil_sensitivity_maps = torch.tensor([[[[1., 1.], [2., 2.], [3., 3.]], [[1., 1.], [2., 2.], [3., 3.]]], \
    [[[1., 1.], [2., 2.], [3., 3.]], [[1., 1.], [2., 2.], [3., 3.]]]])
    >>> coil_sensitivity_maps.shape
    torch.Size([2, 2, 3, 2])
    >>> sense(data, coil_sensitivity_maps)
    tensor([[[2.8284, 2.8284],
        [5.6569, 5.6569],
        [8.4853, 8.4853]],
        [[2.8284, 2.8284],
        [5.6569, 5.6569],
        [8.4853, 8.4853]]])
    >>> sense(data, coil_sensitivity_maps).shape
    torch.Size([2, 3, 2])
    """
    return complex_mul(x, complex_conj(sensitivity_maps)).sum(dim)


def coil_combination_method(
    x: torch.Tensor, sensitivity_maps: torch.Tensor, method: str = "SENSE", dim: int = 0
) -> torch.Tensor:
    """
    Selects the coil combination method.

    Parameters
    ----------
    x : torch.Tensor
        The tensor to coil-combine.
    sensitivity_maps : torch.Tensor
        The coil sensitivity maps.
    method : str, optional
        The coil combination method to use. Options are ``"SENSE"``, ``"RSS"``, ``"RSS_COMPLEX"``.
        Default is ``"SENSE"``.
    dim : int, optional
        The dimension to coil-combine along. Default is ``0``.

    Returns
    -------
    torch.Tensor
        Coil-combined tensor with the selected method applied.

    Examples
    --------
    >>> from mridc.collections.common.parts.utils import coil_combination_method
    >>> import torch
    >>> data = torch.tensor([[[[1., 1.], [2., 2.], [3., 3.]], [[1., 1.], [2., 2.], [3., 3.]]], \
    [[[1., 1.], [2., 2.], [3., 3.]], [[1., 1.], [2., 2.], [3., 3.]]]])
    >>> data.shape
    torch.Size([2, 2, 3, 2])
    >>> coil_sensitivity_maps = torch.tensor([[[[1., 1.], [2., 2.], [3., 3.]], [[1., 1.], [2., 2.], [3., 3.]]], \
    [[[1., 1.], [2., 2.], [3., 3.]], [[1., 1.], [2., 2.], [3., 3.]]]])
    >>> coil_sensitivity_maps.shape
    torch.Size([2, 2, 3, 2])
    >>> coil_combination_method(data, coil_sensitivity_maps, method="SENSE")
    tensor([[[2.8284, 2.8284],
        [5.6569, 5.6569],
        [8.4853, 8.4853]],
        [[2.8284, 2.8284],
        [5.6569, 5.6569],
        [8.4853, 8.4853]]])
    >>> coil_combination_method(data, coil_sensitivity_maps, method="SENSE").shape
    torch.Size([2, 3, 2])
    >>> coil_combination_method(data, coil_sensitivity_maps, method="RSS")
    tensor([[[1.4142, 1.4142],
        [2.8284, 2.8284],
        [4.2426, 4.2426]],
        [[1.4142, 1.4142],
        [2.8284, 2.8284],
        [4.2426, 4.2426]]])
    >>> coil_combination_method(data, coil_sensitivity_maps, method="RSS").shape
    torch.Size([2, 3, 2])
    >>> coil_combination_method(data, coil_sensitivity_maps, method="RSS_COMPLEX")
    tensor([[[1.4142, 1.4142],
        [2.8284, 2.8284],
        [4.2426, 4.2426]],
        [[1.4142, 1.4142],
        [2.8284, 2.8284],
        [4.2426, 4.2426]]])
    >>> coil_combination_method(data, coil_sensitivity_maps, method="RSS_COMPLEX").shape
    torch.Size([2, 3, 2])
    """
    if method == "SENSE":
        return sense(x, sensitivity_maps, dim)
    if method == "RSS":
        return rss(x, dim)
    if method == "RSS_COMPLEX":
        return rss_complex(x, dim)
    raise ValueError("Output type not supported.")


def save_predictions(
    predictions: Dict[str, np.ndarray], out_dir: Path, key: str = "reconstructions", format: str = "h5"
) -> None:
    """
    Save predictions to selected format.

    Parameters
    ----------
    predictions : Dict[str, np.ndarray]
        A dictionary mapping input filenames to corresponding predictions.
    out_dir : Path
        The output directory to save the predictions to.
    key : str, optional
        The key to save the predictions under. Default is ``reconstructions``.
    format : str, optional
        The format to save the predictions in. Default is ``h5``.

    Examples
    --------
    >>> from mridc.collections.common.parts.utils import save_predictions
    >>> import numpy as np
    >>> from pathlib import Path
    >>> data = {"test.h5": np.array([[[1., 1.], [2., 2.], [3., 3.]], [[1., 1.], [2., 2.], [3., 3.]]])}
    >>> data["test.h5"].shape
    (2, 3, 2)
    >>> output_directory = Path("predictions")
    >>> save_predictions(data, output_directory, key="reconstructions", format="h5")
    >>> save_predictions(data, output_directory, key="segmentations", format="h5")
    """
    if format != "h5":
        raise ValueError(f"Output format {format} is not supported.")
    out_dir.mkdir(exist_ok=True, parents=True)
    for fname, preds in predictions.items():
        with h5py.File(out_dir / fname, "w") as hf:
            hf.create_dataset(key, data=preds)


def apply_mask(
    x: torch.Tensor,
    mask_func: MaskFunc,
    seed: Optional[Union[int, Tuple[int, ...]]] = None,
    padding: Optional[Sequence[int]] = None,
    shift: bool = False,
    half_scan_percentage: Optional[float] = 0.0,
    center_scale: Optional[float] = 0.02,
    existing_mask: Optional[torch.Tensor] = None,
) -> Tuple[Any, Any, int]:
    """
    Retrospectively accelerate/subsample k-space data by applying a mask to the input data.

    Parameters
    ----------
    x : torch.Tensor
        The input k-space data. This should have at least 3 dimensions, where dimensions -3 and -2 are the spatial
        dimensions, and the final dimension has size 2 (for complex values).
    mask_func : MaskFunc
        A function that takes a shape (tuple of ints) and a random number seed and returns a mask.
    seed : Optional[Union[int, Tuple[int, ...]]], optional
        Seed for the random number generator. Default is ``None``.
    padding : Optional[Sequence[int]], optional
        Padding value to apply for mask. Default is ``None``.
    shift : bool, optional
        Toggle to shift mask when subsampling. Applicable on 2D data. Default is ``False``.
    half_scan_percentage : Optional[float], optional
        Percentage of kspace to be dropped. Default is ``0.0``.
    center_scale : Optional[float], optional
        Scale of the center of the mask. Applicable on Gaussian masks. Default is ``0.02``.
    existing_mask : Optional[torch.Tensor], optional
        When given, use this mask instead of generating a new one. Default is ``None``.

    Returns
    -------
    Tuple[Any, Any, int]
        Tuple containing the masked k-space data, the mask, and the acceleration factor.

    Examples
    --------
    >>> from mridc.collections.common.parts.utils import apply_mask
    >>> import torch
    >>> data = torch.tensor([[[[1., 1.], [2., 2.], [3., 3.]], [[1., 1.], [2., 2.], [3., 3.]]], \
    [[[1., 1.], [2., 2.], [3., 3.]], [[1., 1.], [2., 2.], [3., 3.]]]])
    >>> data.shape
    torch.Size([2, 2, 3, 2])
    >>> mask = torch.tensor([[[1., 1., 1.], [1., 1., 1.]], [[1., 1., 1.], [1., 1., 1.]]])
    >>> mask.shape
    torch.Size([2, 2, 3])
    >>> apply_mask(data, mask)
    (tensor([[[[1., 1.], [2., 2.], [3., 3.]], [[1., 1.], [2., 2.], [3., 3.]]],
        [[[1., 1.], [2., 2.], [3., 3.]], [[1., 1.], [2., 2.], [3., 3.]]]]),
    tensor([[[1., 1., 1.], [1., 1., 1.]], [[1., 1., 1.], [1., 1., 1.]]]),
    6)
    >>> masked_data, subsampling_mask, acceleration_factor = apply_mask(data, mask)
    >>> masked_data.shape
    torch.Size([2, 2, 3, 2])
    >>> subsampling_mask.shape
    torch.Size([2, 2, 3])
    >>> acceleration_factor
    6
    >>> apply_mask(data, mask, padding=[1, 2], shift=True)
    (tensor([[[[0., 0.], [0., 0.], [0., 0.]], [[1., 1.], [2., 2.], [3., 3.]]],
        [[[0., 0.], [0., 0.], [0., 0.]], [[1., 1.], [2., 2.], [3., 3.]]]]),
    tensor([[[0., 0., 0.], [1., 1., 1.]], [[0., 0., 0.], [1., 1., 1.]]]),
    3)
    >>> masked_data, subsampling_mask, acceleration_factor = apply_mask(data, mask, padding=[1, 2], shift=True)
    >>> masked_data.shape
    torch.Size([2, 2, 3, 2])
    >>> subsampling_mask.shape
    torch.Size([2, 2, 3])
    >>> acceleration_factor
    3
    """
    shape = np.array(x.shape)
    shape[:-3] = 1

    if existing_mask is None:
        mask, acc = mask_func(shape, seed, half_scan_percentage=half_scan_percentage, scale=center_scale)
    else:
        mask = existing_mask
        acc = mask.size / mask.sum()

    mask = mask.to(x.device)

    if padding is not None and padding[0] != 0:
        mask[:, :, : padding[0]] = 0
        # padding value inclusive on right of zeros
        mask[:, :, padding[1] :] = 0

    if shift:
        mask = torch.fft.fftshift(mask, dim=(1, 2))

    masked_x = x * mask + 0.0  # the + 0.0 removes the sign of the zeros

    return masked_x, mask, acc


def mask_center(
    x: torch.Tensor, mask_from: Optional[int], mask_to: Optional[int], mask_type: str = "2D"
) -> torch.Tensor:
    """
    Apply a center crop to the input real image or batch of real images.

    Parameters
    ----------
    x : torch.Tensor
        The input image or batch of images. This should have at least 3 dimensions, where dimensions -3 and -2 are the
        spatial dimensions, and the final dimension has size 1 (for real values).
    mask_from : Optional[int]
        Part of center to start filling.
    mask_to : Optional[int]
        Part of center to end filling.
    mask_type : str, optional
        Type of mask to apply. Can be either ``1D`` or ``2D``. Default is ``2D``.

    Returns
    -------
    torch.Tensor
        The masked image or batch of images with filled center.

    Examples
    --------
    >>> from mridc.collections.common.parts.utils import mask_center
    >>> import torch
    >>> data = torch.tensor([[[[1., 1.], [2., 2.], [3., 3.]], [[1., 1.], [2., 2.], [3., 3.]]], \
    [[[1., 1.], [2., 2.], [3., 3.]], [[1., 1.], [2., 2.], [3., 3.]]]])
    >>> data.shape
    torch.Size([2, 2, 3, 2])
    >>> mask_center(data, 1, 2)
    tensor([[[[0., 0.], [1., 1.], [0., 0.]], [[0., 0.], [1., 1.], [0., 0.]]],
        [[[0., 0.], [1., 1.], [0., 0.]], [[0., 0.], [1., 1.], [0., 0.]]]])
    >>> mask_center(data, 1, 2).shape
    torch.Size([2, 2, 3, 2])
    """
    mask = torch.zeros_like(x)

    if isinstance(mask_from, list):
        mask_from = mask_from[0]

    if isinstance(mask_to, list):
        mask_to = mask_to[0]

    if mask_type == "1D":
        mask[:, :, :, mask_from:mask_to] = x[:, :, :, mask_from:mask_to]
    elif mask_type == "2D":
        mask[:, :, mask_from:mask_to] = x[:, :, mask_from:mask_to]
    else:
        raise ValueError(f"Unknown mask type {mask_type}")

    return mask


def batched_mask_center(
    x: torch.Tensor, mask_from: torch.Tensor, mask_to: torch.Tensor, mask_type: str = "2D"
) -> torch.Tensor:
    """
    Initializes a mask with the center filled in. Can operate with different masks for each batch element.

    Parameters
    ----------
    x : torch.Tensor
        The input image or batch of images. This should have at least 3 dimensions, where dimensions -3 and -2 are the
        spatial dimensions, and the final dimension has size 1 (for real values).
    mask_from : torch.Tensor
        Part of center to start filling.
    mask_to : torch.Tensor
        Part of center to end filling.
    mask_type : str, optional
        Type of mask to apply. Can be either ``1D`` or ``2D``. Default is ``2D``.

    Returns
    -------
    torch.Tensor
        The masked image or batch of images with filled center.

    Examples
    --------
    >>> from mridc.collections.common.parts.utils import batched_mask_center
    >>> import torch
    >>> data = torch.tensor([[[[1., 1.], [2., 2.], [3., 3.]], [[1., 1.], [2., 2.], [3., 3.]]], \
    [[[1., 1.], [2., 2.], [3., 3.]], [[1., 1.], [2., 2.], [3., 3.]]]])
    >>> data.shape
    torch.Size([2, 2, 3, 2])
    >>> batched_mask_center(data, torch.tensor([1, 1]), torch.tensor([2, 2]))
    tensor([[[[0., 0.], [1., 1.], [0., 0.]], [[0., 0.], [1., 1.], [0., 0.]]],
        [[[0., 0.], [1., 1.], [0., 0.]], [[0., 0.], [1., 1.], [0., 0.]]]])
    >>> batched_mask_center(data, torch.tensor([1, 1]), torch.tensor([2, 2])).shape
    torch.Size([2, 2, 3, 2])
    """
    if mask_from.shape != mask_to.shape:
        raise ValueError("mask_from and mask_to must match shapes.")
    if mask_from.ndim != 1:
        raise ValueError("mask_from and mask_to must have 1 dimension.")
    if mask_from.shape[0] not in (1, x.shape[0]) or x.shape[0] != mask_to.shape[0]:
        raise ValueError("mask_from and mask_to must have batch_size length.")

    if mask_from.shape[0] == 1:
        mask = mask_center(x, int(mask_from), int(mask_to), mask_type=mask_type)
    else:
        mask = torch.zeros_like(x)
        for i, (start, end) in enumerate(zip(mask_from, mask_to)):
            mask[i, :, :, start:end] = x[i, :, :, start:end]

    return mask


def center_crop(x: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Apply a center crop to the input complex image or batch of complex images or real image or batch of real images
    without a complex dimension.

    Parameters
    ----------
    x : torch.Tensor
        The input tensor to be center cropped. It should have at least 2 dimensions and the cropping is applied along
        the last two dimensions.
    shape : Tuple[int, int]
        The output shape. The shape should be smaller than the corresponding dimensions of data.

    Returns
    -------
    torch.Tensor
        The center cropped image or batch of images.

    Examples
    --------
    >>> from mridc.collections.common.parts.utils import center_crop
    >>> import torch
    >>> data = torch.tensor([[[1+1j, 2+2j, 3+3j], [1+1j, 2+2j, 3+3j]], [[1+1j, 2+2j, 3+3j], [1+1j, 2+2j, 3+3j]]])
    >>> data.shape
    torch.Size([2, 2, 3])
    >>> center_crop(data, (1, 2))
    tensor([[[2.+2.j, 3.+3.j]], [[2.+2.j, 3.+3.j]]])
    >>> center_crop(data, (1, 2)).shape
    torch.Size([2, 1, 2])
    """
    if not (0 < shape[0] <= x.shape[-2] and 0 < shape[1] <= x.shape[-1]):
        raise ValueError("Invalid shapes.")

    w_from = torch.div((x.shape[-2] - shape[0]), 2, rounding_mode="trunc")
    h_from = torch.div((x.shape[-1] - shape[1]), 2, rounding_mode="trunc")
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return x[..., w_from:w_to, h_from:h_to]  # type: ignore


def complex_center_crop(x: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Apply a center crop to the input image or batch of complex images.

    Parameters
    ----------
    x : torch.Tensor
        The input tensor to be center cropped. It should have at least 3 dimensions and the cropping is applied along
        the last two dimensions.
    shape : Tuple[int, int]
        The output shape. The shape should be smaller than the corresponding dimensions of data.

    Returns
    -------
    torch.Tensor
        The complex center cropped image or batch of images.

    Examples
    --------
    >>> from mridc.collections.common.parts.utils import complex_center_crop
    >>> import torch
    >>> data = torch.tensor([[[[1., 1.], [2., 2.], [3., 3.]], [[1., 1.], [2., 2.], [3., 3.]]], \
    [[[1., 1.], [2., 2.], [3., 3.]], [[1., 1.], [2., 2.], [3., 3.]]]])
    >>> data.shape
    torch.Size([2, 2, 3, 2])
    >>> complex_center_crop(data, (1, 2))
    tensor([[[[2., 2.]]],
        [[[2., 2.]]]])
    >>> complex_center_crop(data, (1, 2)).shape
    torch.Size([2, 1, 1, 2])
    """
    if not (0 < shape[0] <= x.shape[-3] and 0 < shape[1] <= x.shape[-2]):
        raise ValueError("Invalid shapes.")

    w_from = torch.div((x.shape[-3] - shape[0]), 2, rounding_mode="trunc")
    h_from = torch.div((x.shape[-2] - shape[1]), 2, rounding_mode="trunc")
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return x[..., w_from:w_to, h_from:h_to, :]  # type: ignore


def center_crop_to_smallest(
    x: Union[torch.Tensor, np.ndarray], y: Union[torch.Tensor, np.ndarray]
) -> Tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]]:
    """
    Apply a center crop on the larger image to the size of the smaller.

    The minimum is taken over dim=-1 and dim=-2. If x is smaller than y at dim=-1 and y is smaller than x at dim=-2,
        then the returned dimension will be a mixture of the two.

    Parameters
    ----------
    x : torch.Tensor or np.ndarray
        The first image.
    y : torch.Tensor or np.ndarray
        The second image.

    Returns
    -------
    Tuple[torch.Tensor or np.ndarray, torch.Tensor or np.ndarray]
        Tuple of x and y, cropped to the minimum size.

    Examples
    --------
    >>> from mridc.collections.common.parts.utils import center_crop_to_smallest
    >>> import torch
    >>> data1 = torch.tensor([[[1+1j, 2+2j, 3+3j], [1+1j, 2+2j, 3+3j]], [[1+1j, 2+2j, 3+3j], [1+1j, 2+2j, 3+3j]]])
    >>> data2 = torch.tensor([[[1+1j, 2+2j, 3+3j, 4+4j, 5+5j], [1+1j, 2+2j, 3+3j, 4+4j, 5+5j]], \
    [[1+1j, 2+2j, 3+3j, 4+4j, 5+5j], [1+1j, 2+2j, 3+3j, 4+4j, 5+5j], [1+1j, 2+2j, 3+3j, 4+4j, 5+5j]]])
    >>> data1.shape
    torch.Size([2, 2, 3])
    >>> data2.shape
    torch.Size([2, 3, 5])
    >>> center_crop_to_smallest(data1, data2)
    (tensor([[[1+1j, 2+2j, 3+3j], [1+1j, 2+2j, 3+3j]], [[1+1j, 2+2j, 3+3j], [1+1j, 2+2j, 3+3j]]]), \
    tensor([[[1.+1.j, 2.+2.j, 3.+3.j], [1.+1.j, 2.+2.j, 3.+3.j]], \
    [[1.+1.j, 2.+2.j, 3.+3.j], [1.+1.j, 2.+2.j, 3.+3.j]]]))
    >>> center_crop_to_smallest(data1, data2)[0].shape
    torch.Size([2, 2, 3])
    >>> center_crop_to_smallest(data1, data2)[1].shape
    torch.Size([2, 2, 3])
    >>> center_crop_to_smallest(data2, data1)
    (tensor([[[1.+1.j, 2.+2.j, 3.+3.j], [1.+1.j, 2.+2.j, 3.+3.j]], \
    [[1.+1.j, 2.+2.j, 3.+3.j], [1.+1.j, 2.+2.j, 3.+3.j]]]), \
    tensor([[[1+1j, 2+2j, 3+3j], [1+1j, 2+2j, 3+3j]], [[1+1j, 2+2j, 3+3j], [1+1j, 2+2j, 3+3j]]]))
    >>> center_crop_to_smallest(data2, data1)[0].shape
    torch.Size([2, 2, 3])
    >>> center_crop_to_smallest(data2, data1)[1].shape
    torch.Size([2, 2, 3])
    """
    smallest_width = min(x.shape[-1], y.shape[-1])
    smallest_height = min(x.shape[-2], y.shape[-2])
    x = center_crop(x, (smallest_height, smallest_width))
    y = center_crop(y, (smallest_height, smallest_width))

    return x, y


def rnn_weights_init(module: torch.nn.Module, std_init_range: float = 0.02, xavier: bool = True):
    """
    Initialize weights in Recurrent Neural Network.

    Parameters
    ----------
    module : torch.nn.Module
        Module to initialize.
    std_init_range : float
        Standard deviation of normal initializer. Default is ``0.02``.
    xavier : bool
        If True, xavier initializer will be used in Linear layers as in [1].
        Otherwise, normal initializer will be used.
        Default is ``True``.

    References
    ----------
    .. [1] Vaswani A, Shazeer N, Parmar N, Uszkoreit J, Jones L, Gomez AN, Kaiser Ł, Polosukhin I. Attention is all
        you need. Advances in neural information processing systems. 2017;30.

    Examples
    --------
    >>> import torch
    >>> from mridc.collections.common.parts.utils import rnn_weights_init
    >>> rnn = torch.nn.GRU(10, 20, 2)
    >>> rnn.apply(rnn_weights_init)
    GRU(10, 20, num_layers=2)
    """
    if isinstance(module, torch.nn.Linear):
        if xavier:
            torch.nn.init.xavier_uniform_(module.weight)
        else:
            torch.nn.init.normal_(module.weight, mean=0.0, std=std_init_range)
        if module.bias is not None:
            torch.nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, torch.nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=std_init_range)
    elif isinstance(module, torch.nn.LayerNorm):
        torch.nn.init.constant_(module.weight, 1.0)
        torch.nn.init.constant_(module.bias, 0.0)


def add_coil_dim_if_singlecoil(x: torch.tensor, dim: int = 0) -> torch.tensor:
    """
    Add dummy coil dimension if single coil data.

    Parameters
    ----------
    x : torch.tensor
        The input data.
    dim : int
        The dimension to add coil dimension. Default is ``0``.

    Returns
    -------
    torch.tensor
        The input data with coil dimension added if single coil.

    Examples
    --------
    >>> import torch
    >>> from mridc.collections.common.parts.utils import add_coil_dim_if_singlecoil
    >>> data = torch.rand(10, 10)
    >>> data.shape
    (10, 10)
    >>> add_coil_dim_if_singlecoil(data).shape
    (1, 10, 10)
    >>> add_coil_dim_if_singlecoil(data, dim=-1).shape
    (10, 10, 1)
    """
    if len(x.shape) >= 4:
        return x
    else:
        return torch.unsqueeze(x, dim=dim)
