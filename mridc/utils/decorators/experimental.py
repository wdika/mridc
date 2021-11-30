# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/utils/decorators/experimental.py

__all__ = ["experimental"]

from mridc.utils import logging


def experimental(cls):
    """
    Decorator to mark a class as experimental.

    Parameters
    ----------
    cls: The class to be decorated.
        class

    Returns
    -------
    The decorated class.
    """

    def wrapped(x):
        """
        Wrapper function.

        Parameters
        ----------
        x: The class to be decorated.
            class

        Returns
        -------
        The decorated class with the experimental flag set.
        """
        logging.warning(
            f"Module {x} is experimental, not ready for production and is not fully supported. Use at your own risk."
        )

        return x

    return wrapped(x=cls)
