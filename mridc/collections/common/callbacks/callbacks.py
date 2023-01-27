# coding=utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/common/callbacks/callbacks.py

import time

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only


class LogEpochTimeCallback(Callback):
    """
    Simple callback that logs how long each epoch takes, in seconds, to a pytorch lightning log.

    Examples
    --------
    >>> from pytorch_lightning import Trainer
    >>> from pytorch_lightning.callbacks import ModelCheckpoint
    >>> from pytorch_lightning.loggers import TensorBoardLogger
    >>> from mridc.collections.common.callbacks.callbacks import LogEpochTimeCallback

    >>> logger = TensorBoardLogger("tb_logs", name="my_model")
    >>> checkpoint_callback = ModelCheckpoint(
    ...     dirpath="checkpoints",
    ...     filename="my_model-{epoch:02d}-{val_loss:.2f}",
    ...     save_top_k=3,
    ...     verbose=True,
    ...     monitor="val_loss",
    ...     mode="min",
    ... )
    >>> trainer = Trainer(
    ...     logger=logger,
    ...     callbacks=[LogEpochTimeCallback(), checkpoint_callback],
    ...     max_epochs=10,
    ...     gpus=1,
    ...     strategy="ddp_fork",
    ...     accelerator="gpu",
    ...     precision=16,
    ... )
    """

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
