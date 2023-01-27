# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from mridc.utils.app_state import AppState
from mridc.utils.cast_utils import (
    CastToFloat,
    CastToFloatAll,
    avoid_bfloat16_autocast_context,
    avoid_float16_autocast_context,
    cast_all,
    cast_tensor,
)
from mridc.utils.mridc_logging import Logger as _Logger
from mridc.utils.mridc_logging import LogMode as logging_mode

logging = _Logger()
try:
    from mridc.utils.lightning_logger_patch import add_memory_handlers_to_pl_logger

    add_memory_handlers_to_pl_logger()
except ModuleNotFoundError:
    pass
