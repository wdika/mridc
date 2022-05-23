# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/utils/timers.py

import time

import numpy as np
import torch

__all__ = ["NamedTimer"]


class NamedTimer:
    """
    A timer class that supports multiple named timers.
    A named timer can be used multiple times, in which case the average dt will be returned.
    A named timer cannot be started if it is already currently running.
    Use case: measuring execution of multiple code blocks.
    """

    _REDUCTION_TYPE = ["mean", "sum", "min", "max", "none"]

    def __init__(self, reduction="mean", sync_cuda=False, buffer_size=-1):
        """

        Parameters
        ----------
        reduction: Reduction over multiple timings of the same timer (none - returns the list instead of a scalar).
        sync_cuda: If True torch.cuda.synchronize() is called for start/stop
        buffer_size: If positive, limits the number of stored measures per name
        """
        if reduction not in self._REDUCTION_TYPE:
            raise ValueError(f"Unknown reduction={reduction} please use one of {self._REDUCTION_TYPE}")

        self._reduction = reduction
        self._sync_cuda = sync_cuda
        self._buffer_size = buffer_size

        self.reset()

    def __getitem__(self, k):
        return self.get(k)

    @property
    def buffer_size(self):
        """Returns the buffer size of the timer."""
        return self._buffer_size

    @property
    def _reduction_fn(self):
        """Returns the reduction function for the timer."""
        if self._reduction == "none":

            def fn(x):
                return x

        else:
            fn = getattr(np, self._reduction)

        return fn

    def reset(self, name=None):
        """
        Resents all / specific timer

        Parameters
        ----------
        name: Timer name to reset (if None all timers are reset)
        """
        if name is None:
            self.timers = {}
        else:
            self.timers[name] = {}

    def start(self, name=""):
        """
        Starts measuring a named timer.

        Parameters
        ----------
        name: timer name to start
        """
        timer_data = self.timers.get(name, {})

        if "start" in timer_data:
            raise RuntimeError(f"Cannot start timer = '{name}' since it is already active")

        # synchronize pytorch cuda execution if supported
        if self._sync_cuda and torch.cuda.is_initialized():
            torch.cuda.synchronize()

        timer_data["start"] = time.time()

        self.timers[name] = timer_data

    def stop(self, name=""):
        """
        Stops measuring a named timer.

        Parameters
        ----------
        name: timer name to stop
        """
        timer_data = self.timers.get(name)
        if (timer_data is None) or ("start" not in timer_data):
            raise RuntimeError(f"Cannot end timer = '{name}' since it is not active")

        # synchronize pytorch cuda execution if supported
        if self._sync_cuda and torch.cuda.is_initialized():
            torch.cuda.synchronize()

        # compute dt and make timer inactive
        dt = time.time() - timer_data.pop("start")

        # store dt
        timer_data["dt"] = timer_data.get("dt", []) + [dt]

        # enforce buffer_size if positive
        if self._buffer_size > 0:
            timer_data["dt"] = timer_data["dt"][-self._buffer_size :]

        self.timers[name] = timer_data

    def active_timers(self):
        """Return list of all active named timers"""
        return [k for k, v in self.timers.items() if "start" in v]

    def get(self, name=""):
        """
        Returns the value of a named timer

        Parameters
        ----------
        name: timer name to return
        """
        dt_list = self.timers[name].get("dt", [])

        return self._reduction_fn(dt_list)

    def export(self):
        """Exports a dictionary with average/all dt per named timer"""
        fn = self._reduction_fn

        return {k: fn(v["dt"]) for k, v in self.timers.items() if "dt" in v}
