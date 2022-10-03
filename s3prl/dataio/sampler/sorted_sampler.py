"""
The most commonly used batch sampler in S3PRL legacy codebase,
which sorts the lengths of all the data points and group the instances
with the similar lengths together.

Authors:
  Leo 2022
"""

from typing import List

import torch

__all__ = [
    "SortedSliceSampler",
    "SortedBucketingSampler",
]


class SortedSliceSampler:
    """
    This sampler should only be used for training hence is always in random shuffle mode

    Args:
        lengths (List[int])
        batch_size (int): the default batch size
        max_length (int): if a batch contains at least on utt longer than max_length, half the batch
        get_length_func (callable): get the length of each item in the dataset, if None, a default function will be used
        in_batch_shuffle (bool): if False, batches are sorted by length from long to short
    """

    def __init__(
        self,
        lengths: List[int],
        batch_size: int,
        max_length: int = 300000,
        seed: int = 12345678,
        in_batch_shuffle: bool = False,
    ) -> None:
        self.lengths = lengths
        self.epoch = 0
        self.seed = seed
        self.batch_size = batch_size
        self.max_length = max_length
        self.in_batch_shuffle = in_batch_shuffle

        sorted_ids = [(idx, length) for idx, length in enumerate(lengths)]
        sorted_ids = sorted(sorted_ids, key=lambda x: x[1], reverse=True)
        self.sorted_ids = [data_id for data_id, length in sorted_ids]

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.epoch + self.seed)

        indices = torch.randperm(len(self.lengths), generator=generator).tolist()

        for indice in indices:
            length = self.lengths[indice]
            if length > self.max_length:
                batch_size = self.batch_size // 2
            else:
                batch_size = self.batch_size
            start_position = self.sorted_ids.index(indice)
            batch = self.sorted_ids[start_position : start_position + batch_size]

            if self.in_batch_shuffle:
                inbatch_indices = torch.randperm(
                    len(batch), generator=generator
                ).tolist()
                batch = [batch[idx] for idx in inbatch_indices]

            yield batch

    def __len__(self):
        return len(list(iter(self)))


class SortedBucketingSampler:
    """
    Args:
        lengths (List[int])
        batch_size (int): the default batch size
        max_length (int): if a batch contains at least on utt longer than max_length, half the batch
        get_length_func (callable): get the length of each item in the dataset, if None, a default function will be used
        shuffle (bool): Whether to shuffle the batches
        in_batch_shuffle (bool): if False, batches are sorted by length from long to short
    """

    def __init__(
        self,
        lengths: List[int],
        batch_size: int,
        max_length: int = 300000,
        shuffle: bool = False,
        in_batch_shuffle: bool = False,
        seed: int = 12345678,
    ) -> None:
        self.epoch = 0
        self.seed = seed
        self.batch_size = batch_size
        self.max_length = max_length
        self.shuffle = shuffle
        self.in_batch_shuffle = in_batch_shuffle
        self.lengths = lengths

        sorted_ids = [(idx, length) for idx, length in enumerate(self.lengths)]
        sorted_ids = sorted(sorted_ids, key=lambda x: x[1], reverse=True)
        self.sorted_ids = [data_id for data_id, length in sorted_ids]

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.epoch + self.seed)

        batches = []
        position = 0
        while position < len(self.sorted_ids):
            indice = self.sorted_ids[position]
            length = self.lengths[indice]
            if length > self.max_length:
                batch_size = self.batch_size // 2
            else:
                batch_size = self.batch_size
            batch = self.sorted_ids[
                position : min(position + batch_size, len(self.sorted_ids))
            ]
            position += batch_size
            if self.in_batch_shuffle:
                shuffled_batch_indices = torch.randperm(len(batch), generator=generator)
                batch = [batch[idx] for idx in shuffled_batch_indices]
            batches.append(batch)

        if self.shuffle:
            shuffled_indices = torch.randperm(len(batches), generator=generator)
            batches = [batches[idx] for idx in shuffled_indices]

        return iter(batches)

    def __len__(self):
        return len(list(iter(self)))
