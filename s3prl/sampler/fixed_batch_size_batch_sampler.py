from typing import Iterator, TypeVar

from speechbrain.dataio.sampler import ReproducibleRandomSampler
from torch.utils.data import BatchSampler, Sampler, SequentialSampler

from .base import Sampler

T_co = TypeVar("T_co", covariant=True)


class FixedBatchSizeBatchSampler(Sampler):
    """
    The reduced timestamps for a batch should not exceed the max_timestamp.
    If shuffled, each indices are first shuffled before aggregated into batches
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        shuffle: bool = False,
        seed: int = 12345678,
    ) -> None:
        self.batch_size = batch_size

        super().__init__(dataset)
        if shuffle:
            self.sampler = ReproducibleRandomSampler(dataset, seed=seed)
        else:
            self.sampler = SequentialSampler(dataset)

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)

    def _evaluate_reduced_timestamps(self, batch_indices):
        return self.reduce_func([self.timestamps[indice] for indice in batch_indices])

    def __iter__(self) -> Iterator[T_co]:
        batch_sampler = BatchSampler(
            self.sampler, batch_size=self.batch_size, drop_last=False
        )
        return iter(batch_sampler)

    def __len__(self):
        return len(list(iter(self)))
