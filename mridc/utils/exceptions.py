# encoding: utf-8
__author__ = "Dimitrios Karkalousos"


# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/utils/exceptions.py


class MRIDCBaseException(Exception):
    """MRIDC Base Exception. All exceptions created in MRIDC should inherit from this class"""


class LightningNotInstalledException(MRIDCBaseException):
    """Exception for when lightning is not installed"""

    def __init__(self, obj):
        message = (
            f" You are trying to use {obj} without installing all of pytorch_lightning, hydra, and "
            f"omegaconf. Please install those packages before trying to access {obj}."
        )
        super().__init__(message)


class CheckInstall:
    """Class to check if a package is installed."""

    def __init__(self, *args, **kwargs):
        raise LightningNotInstalledException(self)

    def __call__(self, *args, **kwargs):
        raise LightningNotInstalledException(self)

    def __getattr__(self, *args, **kwargs):
        raise LightningNotInstalledException(self)
