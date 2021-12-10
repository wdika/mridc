# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/utils/decorators/experimental.py

__all__ = ["experimental"]

from mridc.utils import logging


def experimental(cls):
    """Decorator which indicates that module is experimental.
    Use it to mark experimental or research modules.
    """

    def wrapped(x):
        logging.warning(
            f"Module {x} is experimental, not ready for production and is not fully supported. Use at your own risk."
        )

        return x

    return wrapped(x=cls)
