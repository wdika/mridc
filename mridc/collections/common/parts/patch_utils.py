# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/common/parts/patch_utils.py

from packaging import version

# Library version globals
TORCH_VERSION = None
TORCH_VERSION_MIN = version.Version("1.9.0")
