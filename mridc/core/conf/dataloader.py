# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/core/config/pytorch.py

from dataclasses import dataclass
from typing import Any, Optional

from omegaconf import MISSING

__all__ = ["DataLoaderConfig"]


@dataclass
class DataLoaderConfig:
    """
    Configuration of PyTorch DataLoader.

    ..note:
        For the details on the function/meanings of the arguments, please refer to:
        https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    """

    batch_size: int = MISSING
    shuffle: bool = False
    sampler: Optional[Any] = None
    batch_sampler: Optional[Any] = None
    num_workers: int = 0
    collate_fn: Optional[Any] = None
    pin_memory: bool = False
    drop_last: bool = False
    timeout: int = 0
    worker_init_fn: Optional[Any] = None
    multiprocessing_context: Optional[Any] = None
