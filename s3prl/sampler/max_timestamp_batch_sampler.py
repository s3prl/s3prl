from typing import Iterator, TypeVar

import torch
from torch.utils.data import Sampler

from s3prl.dataset import metadata_mode

T_co = TypeVar("T_co", covariant=True)


class MaxTimestampBatchSampler(Sampler):
    """
    The reduced timestamps for a batch should not exceed the max_timestamp.
    If shuffled, each indices are first shuffled before aggregated into batches
    """

    def __init__(
        self,
        dataset,
        max_timestamp: int,
        shuffle: bool = False,
        seed: int = 12345678,
        reduce_func: callable = None,
    ) -> None:
        with metadata_mode():
            timestamps = [item.timestamp for item in dataset]

        super().__init__(timestamps)
        self.timestamps = timestamps
        self.max_timestamp = max_timestamp
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        def default_reduce_func(timestamps):
            return max(timestamps) * len(timestamps)

        self.reduce_func = reduce_func or default_reduce_func

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _evaluate_reduced_timestamps(self, batch_indices):
        return self.reduce_func([self.timestamps[indice] for indice in batch_indices])

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.epoch + self.seed)
            indices = torch.randperm(len(self.timestamps), generator=generator).tolist()
        else:
            indices = list(range(len(self.timestamps)))

        batch = []
        for indice in indices:
            try_new_batch = batch + [indice]
            if self._evaluate_reduced_timestamps(try_new_batch) <= self.max_timestamp:
                batch = try_new_batch
            elif len(batch) == 0:
                raise ValueError(
                    f"There is a single timestamp {self.timestamps[indice]} larger than "
                    f"max_timestamp {self.max_timestamp}. Please increase "
                    "the max_timestamp."
                )
            else:
                yield batch
                batch = [indice]

        if len(batch) > 0:
            yield batch

    def __len__(self):
        return len(list(iter(self)))
