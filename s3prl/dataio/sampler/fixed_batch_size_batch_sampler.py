"""
The most commonly used batch sampler, recover the default batch sampler used
in torch DataLoader

Authors:
  * Leo 2022
"""

import torch
from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler

__all__ = ["FixedBatchSizeBatchSampler"]


class FixedBatchSizeBatchSampler:
    """
    The reduced timestamps for a batch should not exceed the max_timestamp.
    If shuffled, each indices are first shuffled before aggregated into batches

    Args:
        data_source: __len__ is implemented
    """

    def __init__(
        self,
        data_source,
        batch_size: int,
        shuffle: bool = False,
        seed: int = 12345678,
    ) -> None:
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle

        if shuffle:
            self.generator = torch.Generator()
            self.sampler = RandomSampler(data_source, generator=self.generator)
        else:
            self.sampler = SequentialSampler(data_source)

    def set_epoch(self, epoch: int) -> None:
        if self.shuffle:
            self.generator.manual_seed(self.seed + epoch)

    def _evaluate_reduced_timestamps(self, batch_indices):
        return self.reduce_func([self.timestamps[indice] for indice in batch_indices])

    def __iter__(self):
        batch_sampler = BatchSampler(
            self.sampler, batch_size=self.batch_size, drop_last=False
        )
        return iter(batch_sampler)

    def __len__(self):
        return len(list(iter(self)))
