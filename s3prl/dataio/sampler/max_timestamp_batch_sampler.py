"""
Limit the maximum timestamps in a batch to realize dynamic batching.

Authors:
  * Leo 2022
"""

from typing import List

import torch

__all__ = [
    "MaxTimestampBatchSampler",
]


class MaxTimestampBatchSampler:
    """
    The reduced timestamps for a batch should not exceed the max_timestamp.
    If shuffled, each indices are first shuffled before aggregated into batches
    """

    def __init__(
        self,
        lengths: List[int],
        max_length: int,
        shuffle: bool = False,
        seed: int = 12345678,
        reduce_func: callable = None,
    ) -> None:
        self.lengths = lengths
        self.max_length = max_length
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.reduce_func = reduce_func or self._default_reduce_func

    @staticmethod
    def _default_reduce_func(timestamps):
        return max(timestamps) * len(timestamps)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _evaluate_reduced_timestamps(self, batch_indices):
        return self.reduce_func([self.lengths[indice] for indice in batch_indices])

    def __iter__(self):
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.epoch + self.seed)
            indices = torch.randperm(len(self.lengths), generator=generator).tolist()
        else:
            indices = list(range(len(self.lengths)))

        batch = []
        for indice in indices:
            try_new_batch = batch + [indice]
            if self._evaluate_reduced_timestamps(try_new_batch) <= self.max_length:
                batch = try_new_batch
            elif len(batch) == 0:
                raise ValueError(
                    f"There is a single length {self.lengths[indice]} larger than "
                    f"max_length {self.max_length}. Please increase "
                    "the max_length."
                )
            else:
                yield batch
                batch = [indice]

        if len(batch) > 0:
            yield batch

    def __len__(self):
        return len(list(iter(self)))
