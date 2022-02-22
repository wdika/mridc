# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/utils/lightning_logger_patch.py

import logging as _logging
from logging.handlers import MemoryHandler
from typing import Any, Dict

import pytorch_lightning as pl

HANDLERS: Dict[Any, Any] = {}
PATCHED = False


def add_memory_handlers_to_pl_logger():
    """
    Adds two MemoryHandlers to pytorch_lightning's logger. These two handlers are essentially message buffers. This
    function is called in mridc.utils.__init__.py. These handlers are used in add_filehandlers_to_pl_logger to flush
    buffered messages to files.
    """
    if not HANDLERS:
        HANDLERS["memory_err"] = MemoryHandler(-1)
        HANDLERS["memory_err"].addFilter(lambda record: record.levelno > _logging.INFO)
        HANDLERS["memory_all"] = MemoryHandler(-1)
        pl._logger.addHandler(HANDLERS["memory_err"])
        pl._logger.addHandler(HANDLERS["memory_all"])


def add_filehandlers_to_pl_logger(all_log_file, err_log_file):
    """
    Adds two filehandlers to pytorch_lightning's logger. Called in mridc.utils.exp_manager(). The first filehandler
    logs all messages to all_log_file while the second filehandler logs all WARNING and higher messages to
    err_log_file. If "memory_err" and "memory_all" exist in HANDLERS, then those buffers are flushed to err_log_file
    and all_log_file respectively, and then closed.
    """
    HANDLERS["file"] = _logging.FileHandler(all_log_file)
    pl._logger.addHandler(HANDLERS["file"])
    HANDLERS["file_err"] = _logging.FileHandler(err_log_file)
    HANDLERS["file_err"].addFilter(lambda record: record.levelno > _logging.INFO)
    pl._logger.addHandler(HANDLERS["file_err"])

    if HANDLERS.get("memory_all"):
        HANDLERS["memory_all"].setTarget(HANDLERS["file"])
        HANDLERS["memory_all"].close()
        del HANDLERS["memory_all"]
    if HANDLERS.get("memory_err"):
        HANDLERS["memory_err"].setTarget(HANDLERS["file_err"])
        HANDLERS["memory_err"].close()
        del HANDLERS["memory_err"]
