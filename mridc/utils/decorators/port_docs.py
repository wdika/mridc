# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/utils/decorators/port_docs.py

# The "add_port_docs" decorator is needed to nicely generate neural types in Sphynx for input and output ports

__all__ = ["add_port_docs"]

import functools
import sys

import wrapt


def _normalize_docstring(docstring):
    """
    Normalize docstring indentation. Replaces tabs with spaces, removes leading and trailing blanks lines, and removes
     any indentation.

    Copied from PEP-257: https://www.python.org/dev/peps/pep-0257/#handling-docstring-indentation

    Parameters
    ----------
    docstring: The docstring to normalize.
        str

    Returns
    -------
    The normalized docstring.
    """
    if not docstring:
        return ""
    # Convert tabs to spaces (following the normal Python rules)
    # and split into a list of lines:
    lines = docstring.expandtabs().splitlines()
    # Determine minimum indentation (first line doesn't count):
    # (we use sys.maxsize because sys.maxint doesn't exist in Python 3)
    indent = sys.maxsize
    for line in lines[1:]:
        if stripped := line.lstrip():
            indent = min(indent, len(line) - len(stripped))
    # Remove indentation (first line is special):
    trimmed = [lines[0].strip()]
    if indent < sys.maxsize:
        trimmed.extend(line[indent:].rstrip() for line in lines[1:])
    # Strip off trailing and leading blank lines:
    while trimmed and not trimmed[-1]:
        trimmed.pop()
    while trimmed and not trimmed[0]:
        trimmed.pop(0)
    # Return a single string:
    return "\n".join(trimmed)


def add_port_docs(wrapped=None, instance=None, value=""):
    """
    Adds port documentation to the wrapped function.

    Parameters
    ----------
    wrapped: The function to decorate.
        function
    instance: The instance of the function.
        object
    value: The value of the port.
        object

    Returns
    -------
    The decorated function.
    """
    if wrapped is None:
        return functools.partial(add_port_docs, value=value)

    @wrapt.decorator
    def wrapper(wrapped, instance=None, args=None, kwargs=None):
        """
        Adds port documentation to the wrapped function.

        Parameters
        ----------
        wrapped: The function to decorate.
        instance: The instance of the function.
        args: The arguments of the function.
        kwargs: The keyword arguments of the function.

        Returns
        -------
        The decorated function.
        """
        return wrapped(*args, **kwargs)

    decorated = wrapper(wrapped)
    try:
        port_2_ntype = decorated(instance)
    except AttributeError:
        port_2_ntype = None

    port_description = ""
    if port_2_ntype is not None:
        for port, ntype in port_2_ntype.items():
            port_description += "* *" + port + "* : " + str(ntype)
            port_description += "\n\n"

    __doc__ = _normalize_docstring(wrapped.__doc__) + "\n\n" + str(port_description)
    __doc__ = _normalize_docstring(__doc__)

    wrapt.FunctionWrapper.__setattr__(decorated, "__doc__", __doc__)

    return decorated
