# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/core/classes/dataset.py
from abc import ABC
from dataclasses import dataclass
from typing import Optional

from torch.utils import data

__all__ = ["Dataset", "DatasetConfig", "IterableDataset"]

from mridc.core.classes.common import Serialization, Typing, typecheck


class Dataset(data.Dataset, Typing, Serialization, ABC):
    """Dataset with output ports. Please Note: Subclasses of IterableDataset should *not* implement input_types."""

    @staticmethod
    def _collate_fn(batch):
        """
        A default implementation of a collation function.
        Users should override this method to define custom data loaders.
        """
        return data.dataloader.default_collate(batch)

    @typecheck()
    def collate_fn(self, batch):
        """
        This is the method that user pass as functor to DataLoader.
        The method optionally performs neural type checking and add types to the outputs.

        Please note, subclasses of Dataset should not implement `input_types`.

        # Usage:

        .. code-block::

            dataloader = torch.utils.data.DataLoader(
                    ....,
                    collate_fn=dataset.collate_fn,
                    ....
            )

        Returns
        -------
        Collated batch, with or without types.
        """
        if self.input_types is not None:
            raise TypeError("Datasets should not implement `input_types` as they are not checked")

        # Simply forward the inner `_collate_fn`
        return self._collate_fn(batch)


class IterableDataset(data.IterableDataset, Typing, Serialization, ABC):
    """
    Iterable Dataset with output ports.
    Please Note: Subclasses of IterableDataset should *not* implement input_types.
    """

    @staticmethod
    def _collate_fn(batch):
        """
        A default implementation of a collation function.
        Users should override this method to define custom data loaders.
        """
        return data.dataloader.default_collate(batch)

    @typecheck()
    def collate_fn(self, batch):
        """
        This is the method that user pass as functor to DataLoader.
        The method optionally performs neural type checking and add types to the outputs.

        # Usage:

        .. code-block::

            dataloader = torch.utils.data.DataLoader(
                    ....,
                    collate_fn=dataset.collate_fn,
                    ....
            )

        Returns
        -------
        Collated batch, with or without types.
        """
        if self.input_types is not None:
            raise TypeError("Datasets should not implement `input_types` as they are not checked")

        # Simply forward the inner `_collate_fn`
        return self._collate_fn(batch)


@dataclass
class DatasetConfig:
    """Dataset configuration."""

    batch_size: int = 32
    drop_last: bool = False
    shuffle: bool = False
    num_workers: Optional[int] = 0
    pin_memory: bool = True
