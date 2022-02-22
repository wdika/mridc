# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/common/callbacks/callbacks.py

import time

from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities import rank_zero_only


class LogEpochTimeCallback(Callback):
    """Simple callback that logs how long each epoch takes, in seconds, to a pytorch lightning log"""

    def __init__(self):
        """Initialize the callback."""
        super().__init__()
        self.epoch_start = time.time()

    @rank_zero_only
    def on_train_epoch_start(self, trainer, pl_module):
        """Called at the start of each epoch."""
        self.epoch_start = time.time()

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        """Called at the end of each epoch."""
        curr_time = time.time()
        duration = curr_time - self.epoch_start
        trainer.logger.log_metrics({"epoch_time": duration}, step=trainer.global_step)
