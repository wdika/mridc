# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/common/data/dataset.py

from abc import ABC
from typing import Any, List

import numpy as np
import torch.utils.data as pt_data

__all__ = ["ConcatDataset"]


class ConcatDataset(pt_data.IterableDataset, ABC):
    """
    A dataset that accepts as argument multiple datasets and then samples from them based on the specified
    sampling technique.

    Parameters
    ----------
    datasets: A list of datasets to sample from.
    shuffle: Whether to shuffle individual datasets. Only works with non-iterable datasets. Defaults to True.
    sampling_technique: Sampling technique to choose which dataset to draw a sample from. Defaults to 'random'.
    Currently supports 'random' and 'round-robin'.
    sampling_probabilities: Probability values for sampling. Only used when sampling_technique = 'random'.
    global_rank: Worker rank, used for partitioning map style datasets. Defaults to 0.
    world_size: Total number of processes, used for partitioning map style datasets. Defaults to 1.
    """

    def __init__(
        self,
        datasets: List[Any],
        shuffle: bool = True,
        sampling_technique: str = "random",
        sampling_probabilities: List[float] = None,
        global_rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__()

        self.datasets = datasets
        self.iterables = [None] * len(datasets)
        self.shuffle = shuffle
        self.global_rank = global_rank
        self.world_size = world_size
        self.sampling_kwargs = {}
        if sampling_technique == "random":
            self.index_generator = ConcatDataset.random_generator
            self.sampling_kwargs["p"] = sampling_probabilities  # type: ignore
        elif sampling_technique == "round-robin":
            self.index_generator = ConcatDataset.round_robin_generator
        else:
            supported_sampling_techniques = ["random", "round-robin"]
            raise ValueError(f"Currently we only support sampling techniques in {supported_sampling_techniques}.")
        self.length = 0

        if isinstance(datasets[0], pt_data.IterableDataset):
            self.kind = "iterable"
        else:
            self.kind = "map"

        for dataset in datasets:
            isiterable = isinstance(dataset, pt_data.IterableDataset)
            if isiterable and self.kind != "iterable" or (not isiterable and self.kind == "iterable"):
                raise ValueError("All datasets in ConcatDataset must be of the same kind (Iterable or Map).")

            if self.kind == "map":
                self.length += np.floor_divide(len(dataset), world_size)
            else:
                self.length += len(dataset)

    def get_iterable(self, dataset):
        """Returns an iterable dataset."""
        if isinstance(dataset, pt_data.IterableDataset):
            return dataset.__iter__()
        indices = np.arange(len(dataset))
        if self.shuffle:
            np.random.shuffle(indices)
        return iter(indices)

    def __iter__(self):
        """Returns an iterator over the dataset."""
        worker_info = pt_data.get_worker_info()
        if worker_info is None:
            max_elements = self.length
            wid = 0
            wnum = 1
        else:
            wid = worker_info.id
            wnum = worker_info.num_workers
            max_elements = len(range(wid, self.length, wnum))

        if self.kind == "map":
            for idx in range(len(self.datasets)):
                start_idx = np.floor_divide(len(self.datasets[idx]), self.world_size) * self.global_rank
                end_idx = start_idx + np.floor_divide(len(self.datasets[idx]), self.world_size)
                if self.global_rank == self.world_size - 1:
                    end_idx = len(self.datasets[idx])
                indices = range(start_idx + wid, end_idx, wnum)
                self.datasets[idx] = pt_data.Subset(self.datasets[idx], indices)

        for idx, dataset in enumerate(self.datasets):
            iterable = self.get_iterable(dataset)
            self.iterables[idx] = iterable  # type: ignore

        n = 0
        ind_gen = self.index_generator(self.datasets, **self.sampling_kwargs)
        while n < max_elements:
            n += 1
            try:
                ind = next(ind_gen)
            except StopIteration:
                return
            try:
                val = next(self.iterables[ind])  # type: ignore
                if self.kind == "map":
                    val = self.datasets[ind][val]
                yield val
            except StopIteration:
                self.iterables[ind] = self.get_iterable(self.datasets[ind])  # type: ignore
                n -= 1

    def __len__(self):
        """Returns the number of elements in the dataset."""
        return self.length

    @staticmethod
    def round_robin_generator(datasets, **kwargs):
        """Generates indices in a round-robin fashion."""
        num = len(datasets)
        while True:
            yield from range(num)

    @staticmethod
    def random_generator(datasets, **kwargs):
        """Generates random indices."""
        p = kwargs.get("p")
        if not p:
            raise ValueError("Random generator expects a 'p' keyowrd argument for sampling probabilities.")

        num = len(datasets)
        if len(p) != num:
            raise ValueError("Length of probabilities list must be equal to the number of datasets.")

        while True:
            yield np.random.choice(np.arange(num), p=p)
