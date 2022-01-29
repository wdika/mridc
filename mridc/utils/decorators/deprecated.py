# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/utils/decorators/deprecated.py

__all__ = ["deprecated"]

import functools
from typing import Dict

import wrapt

# Remember which deprecation warnings have been printed already.
from mridc.utils import logging

_PRINTED_WARNING: Dict = {}


def deprecated(wrapped=None, version=None, explanation=None):
    """
    This is a decorator which can be used to mark functions as deprecated. It will result in a warning being emitted
    when the function is used.

    Args:
        wrapped (): The function to be decorated.
        version (str): The version in which the function will be marked as deprecated.
        explanation (str): The explanation of the deprecation.

    Returns:
        The decorated function.
    """
    if wrapped is None:
        return functools.partial(deprecated, version=version, explanation=explanation)

    @wrapt.decorator
    def wrapper(_wrapped, args, kwargs):
        """
        Method prints the adequate warning (only once per function) when required and calls the function func,
        passing the original arguments, i.e. version and explanation.

        Args:
            _wrapped (): The function to be decorated.
            args (): The arguments of the function to be decorated.
            kwargs (): The keyword arguments of the function to be decorated.

        Returns:
            The decorated function.
        """
        # Check if we already warned about that function.
        if _wrapped.__name__ not in _PRINTED_WARNING:
            # Add to list so we won't print it again.
            _PRINTED_WARNING[_wrapped.__name__] = True

            # Prepare the warning message.
            msg = "Function ``{}`` is deprecated.".format(_wrapped.__name__)

            # Optionally, add version and alternative.
            if version is not None:
                msg = msg + " It is going to be removed in "
                msg = msg + "the {} version.".format(version)

            if explanation is not None:
                msg = msg + " " + explanation

            # Display the deprecated warning.
            logging.warning(msg)

        # Call the function.
        return _wrapped(*args, **kwargs)

    return wrapper(wrapped)
