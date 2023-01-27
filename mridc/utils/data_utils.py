# coding=utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/tree/main/nemo/utils/data_utils.py


import os
import pathlib

from mridc import __version__ as MRIDC_VERSION
from mridc import constants


def resolve_cache_dir() -> pathlib.Path:
    """
    Utility method to resolve a cache directory for MRIDC that can be overriden by an environment variable.

    Examples
    --------
        MRIDC_CACHE_DIR="~/mridc_cache_dir/" python mridc_example_script.py

    Returns
    -------
        A Path object, resolved to the absolute path of the cache directory. If no override is provided, uses an
        inbuilt default which adapts to mridc versions strings.
    """
    override_dir = os.environ.get(constants.MRIDC_ENV_CACHE_DIR, "")
    if override_dir == "":
        path = pathlib.Path.joinpath(pathlib.Path.home(), f".cache/torch/MRIDC/MRIDC_{MRIDC_VERSION}")
    else:
        path = pathlib.Path(override_dir).resolve()
    return path
