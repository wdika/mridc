# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/utils/formatters/utils.py

import sys

__all__ = ["check_color_support", "to_unicode"]

from mridc.constants import MRIDC_ENV_VARNAME_ENABLE_COLORING
from mridc.utils.env_var_parsing import get_envbool


def check_color_support():
    # Colors can be forced with an env variable
    if not sys.platform.lower().startswith("win") and get_envbool(MRIDC_ENV_VARNAME_ENABLE_COLORING, False):
        return True


def to_unicode(value):
    """
    Converts a string argument to a unicode string.
    If the argument is already a unicode string or None, it is returned
    unchanged.  Otherwise it must be a byte string and is decoded as utf8.
    """
    try:
        if isinstance(value, (str, type(None))):
            return value

        if not isinstance(value, bytes):
            raise TypeError("Expected bytes, unicode, or None; got %r" % type(value))

        return value.decode("utf-8")

    except UnicodeDecodeError:
        return repr(value)
