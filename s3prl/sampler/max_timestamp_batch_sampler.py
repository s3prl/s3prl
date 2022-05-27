from typing import Iterator, TypeVar

import torch
from joblib import Parallel, delayed
from speechbrain.dataio.dataset import DynamicItemDataset
from torch.utils.data import Sampler
from tqdm import tqdm

from s3prl.dataset import Dataset, metadata_mode

from .base import Sampler

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
        get_timestamps_func: callable = None,
        reduce_func: callable = None,
        n_jobs: int = 4,
    ) -> None:
        if get_timestamps_func is None:
            if isinstance(dataset, Dataset):
                get_timestamps_func = self._get_timestamps_original
            elif isinstance(dataset, DynamicItemDataset):
                get_timestamps_func = self._get_timestamps_dynamic_item_dataset
        timestamps = get_timestamps_func(dataset, n_jobs)

        super().__init__(timestamps)
        self.timestamps = timestamps
        self.max_timestamp = max_timestamp
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.reduce_func = reduce_func or self._default_reduce_func

    @staticmethod
    def _default_reduce_func(timestamps):
        return max(timestamps) * len(timestamps)

    @staticmethod
    def _get_timestamps_original(dataset: Dataset, n_jobs: int = 4):
        with metadata_mode():
            timestamps = [item.timestamp for item in dataset]
        return timestamps

    @staticmethod
    def _get_timestamps_dynamic_item_dataset(
        dataset: DynamicItemDataset, n_jobs: int = 4
    ):
        with dataset.output_keys_as(["wav_metadata"]):

            def get_timestamp(item):
                return item["wav_metadata"]["num_frames"]

            timestamps = Parallel(n_jobs=n_jobs)(
                delayed(get_timestamp)(item)
                for item in tqdm(dataset, desc="loading metadata")
            )
        return timestamps

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
