# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from mridc.utils.mridc_logging import Logger as _Logger

logging = _Logger()
try:
    from mridc.utils.lightning_logger_patch import add_memory_handlers_to_pl_logger

    add_memory_handlers_to_pl_logger()
except ModuleNotFoundError:
    pass
