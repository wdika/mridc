# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/utils/formatters/utils.py

import sys

__all__ = ["check_color_support", "to_unicode"]

from mridc.constants import MRIDC_ENV_VARNAME_ENABLE_COLORING
from mridc.utils.env_var_parsing import get_envbool


def check_color_support():
    """

    Returns
    -------
    True if the terminal supports color, False otherwise.
        bool
    """
    # Colors can be forced with an env variable
    return bool(not sys.platform.lower().startswith("win") and get_envbool(MRIDC_ENV_VARNAME_ENABLE_COLORING, False))


def to_unicode(value):
    """
    Converts a string to unicode. If the string is already unicode, it is returned as is. If it is a byte string, it is
    decoded using utf-8.

    Parameters
    ----------
    value: The string to convert.
        str

    Returns
    -------
    The converted string.
        str
    """
    try:
        if isinstance(value, (str, type(None))):
            return value

        if not isinstance(value, bytes):
            raise TypeError("Expected bytes, unicode, or None; got %r" % type(value))

        return value.decode("utf-8")

    except UnicodeDecodeError:
        return repr(value)
