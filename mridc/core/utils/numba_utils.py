# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/core/utils/numba_utils.py

import contextlib
import logging as pylogger
import operator
import os

# Prevent Numba CUDA logs from showing at info level
from mridc.utils.model_utils import check_lib_version

cuda_logger = pylogger.getLogger("numba.cuda.cudadrv.driver")
cuda_logger.setLevel(pylogger.ERROR)  # only show error

__NUMBA_DEFAULT_MINIMUM_VERSION__ = "0.53.0"
__NUMBA_MINIMUM_VERSION__ = os.environ.get("MRIDC_NUMBA_MINVER", __NUMBA_DEFAULT_MINIMUM_VERSION__)

NUMBA_INSTALLATION_MESSAGE = (
    "Could not import `numba`.\n"
    "Please install numba in one of the following ways."
    "1) If using conda, simply install it with conda using `conda install -c numba numba`\n"
    "2) If using pip (not recommended), `pip install --upgrade numba`\n"
    "followed by `export NUMBAPRO_LIBDEVICE='/usr/local/cuda/nvvm/libdevice/'` and \n"
    "`export NUMBAPRO_NVVM='/usr/local/cuda/nvvm/lib64/libnvvm.so'`.\n"
    "It is advised to always install numba using conda only, "
    "as pip installations might interfere with other libraries such as llvmlite.\n"
    "If pip install does not work, you can also try adding `--ignore-installed` to the pip command,\n"
    "but this is not advised."
)

STRICT_NUMBA_COMPAT_CHECK = True

# Get environment key if available
if "STRICT_NUMBA_COMPAT_CHECK" in os.environ:
    check_str = os.environ.get("STRICT_NUMBA_COMPAT_CHECK")
    check_bool = str(check_str).lower() in {"yes", "true", "t", "1"}
    STRICT_NUMBA_COMPAT_CHECK = check_bool


def is_numba_compat_strict() -> bool:
    """
    Returns strictness level of numba cuda compatibility checks.
    If value is true, numba cuda compatibility matrix must be satisfied.
    If value is false, only cuda availability is checked, not compatibility.
    Numba Cuda may still compile and run without issues in such a case, or it may fail.
    """
    return STRICT_NUMBA_COMPAT_CHECK


def set_numba_compat_strictness(strict: bool):
    """
    Sets the strictness level of numba cuda compatibility checks.
    If value is true, numba cuda compatibility matrix must be satisfied.
    If value is false, only cuda availability is checked, not compatibility.
    Numba Cuda may still compile and run without issues in such a case, or it may fail.

    Parameters
    ----------
    strict: Whether to enforce strict compatibility checks or relax them.
    """
    global STRICT_NUMBA_COMPAT_CHECK
    STRICT_NUMBA_COMPAT_CHECK = strict


@contextlib.contextmanager
def with_numba_compat_strictness(strict: bool):
    """Context manager to temporarily set numba cuda compatibility strictness."""
    initial_strictness = is_numba_compat_strict()
    set_numba_compat_strictness(strict=strict)
    yield
    set_numba_compat_strictness(strict=initial_strictness)


def numba_cpu_is_supported(min_version: str) -> bool:
    """
    Tests if an appropriate version of numba is installed.

    Parameters
    ----------
    min_version: The minimum version of numba that is required.

    Returns
    -------
    bool, whether numba CPU supported with this current installation or not.
    """
    module_available, _ = check_lib_version("numba", checked_version=min_version, operator=operator.ge)

    # If numba is not installed
    if module_available is None:
        return False
    return True


def numba_cuda_is_supported(min_version: str) -> bool:
    """
    Tests if an appropriate version of numba is installed, and if it is,
    if cuda is supported properly within it.

    Parameters
    ----------
    min_version: The minimum version of numba that is required.

    Returns
    -------
    Whether cuda is supported with this current installation or not.
    """
    module_available = numba_cpu_is_supported(min_version)

    # If numba is not installed
    if module_available is None:
        return False

    if module_available is not True:
        return False
    from numba import cuda

    if not hasattr(cuda, "is_supported_version"):
        # assume cuda is supported, but it may fail due to CUDA incompatibility
        return False

    try:
        cuda_available = cuda.is_available()
        cuda_compatible = cuda.is_supported_version() if cuda_available else False
        if is_numba_compat_strict():
            return cuda_available and cuda_compatible
        return cuda_available

    except OSError:
        # dlopen(libcudart.dylib) might fail if CUDA was never installed in the first place.
        return False


def skip_numba_cuda_test_if_unsupported(min_version: str):
    """
    Helper method to skip pytest test case if numba cuda is not supported.

    Parameters
    ----------
    min_version: The minimum version of numba that is required.
    """
    numba_cuda_support = numba_cuda_is_supported(min_version)
    if not numba_cuda_support:
        import pytest

        pytest.skip(f"Numba cuda test is being skipped. Minimum version required : {min_version}")
