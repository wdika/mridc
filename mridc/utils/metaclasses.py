# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/utils/metaclasses.py

import threading
from typing import Any, Dict


class Singleton(type):
    """Implementation of a generic, tread-safe singleton meta-class. Can be used as meta-class, i.e. will create."""

    # List of instances - one per class.
    __instances: Dict[Any, Any] = {}
    # Lock used for accessing the instance.
    __lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        """Returns singleton instance. A thread safe implementation."""
        if cls not in cls.__instances:
            # Enter critical section.
            with cls.__lock:
                # Check once again.
                if cls not in cls.__instances:
                    # Create a new object instance - one per class.
                    cls.__instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        # Return the instance.
        return cls.__instances[cls]
