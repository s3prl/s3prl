"""
For datasets with highly unbalanced class

Authors:
  * Leo 2022
"""

from collections import Counter
from typing import Iterator, List, TypeVar

import torch
from torch.utils.data import WeightedRandomSampler

T_co = TypeVar("T_co", covariant=True)

__all__ = ["BalancedWeightedSampler"]


class BalancedWeightedSampler:
    """
    This batch sampler is always randomized, hence cannot be used for testing
    """

    def __init__(
        self,
        labels: List[str],
        batch_size: int,
        duplicate: int = 1,
        seed: int = 12345678,
    ) -> None:
        self.epoch = 0
        self.seed = seed
        self.batch_size = batch_size
        self.duplicate = duplicate

        class2weight = Counter()
        for label in labels:
            class2weight.update([label])

        weights = []
        for label in labels:
            weights.append(len(labels) / class2weight[label])

        self.weights = weights

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self) -> Iterator[T_co]:
        generator = torch.Generator()
        generator.manual_seed(self.epoch + self.seed)

        sampler = WeightedRandomSampler(
            self.weights, len(self.weights) * self.duplicate, generator=generator
        )
        indices = list(sampler)

        batch = []
        for indice in indices:
            batch.append(indice)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if len(batch) > 0:
            yield batch

    def __len__(self):
        return len(list(iter(self)))
