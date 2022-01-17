# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/core/classes/module.py
from abc import ABC
from contextlib import contextmanager

from torch.nn import Module

__all__ = ["NeuralModule"]

from mridc.core.classes.common import FileIO, Serialization, Typing


class NeuralModule(Module, Typing, Serialization, FileIO, ABC):
    """Abstract class offering interface shared between all PyTorch Neural Modules."""

    @property
    def num_weights(self):
        """Utility property that returns the total number of parameters of NeuralModule."""
        num: int = 0
        for p in self.parameters():
            if p.requires_grad:
                num += p.numel()
        return num

    def input_example(self, max_batch=None, max_dim=None):
        """
        Override this method if random inputs won't work

        Args:
            max_batch: Maximum batch size to generate
            max_dim: Maximum dimension to generate

        Returns:
            A tuple sample of valid input data.
        """
        return None

    def freeze(self) -> None:
        r"""Freeze all params for inference."""
        for param in self.parameters():
            param.requires_grad = False

        self.eval()

    def unfreeze(self) -> None:
        """Unfreeze all parameters for training."""
        for param in self.parameters():
            param.requires_grad = True

        self.train()

    @contextmanager
    def as_frozen(self):
        """Context manager which temporarily freezes a module, yields control and finally unfreezes the module."""
        self.freeze()

        try:
            yield
        finally:
            self.unfreeze()
