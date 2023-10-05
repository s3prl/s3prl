"""
Limit the maximum timestamps in a batch to realize dynamic batching.

Authors:
  * Shu-wen Yang 2022
"""

import torch
from joblib import Parallel, delayed
from tqdm import tqdm

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
        dataset,
        max_timestamp: int,
        shuffle: bool = False,
        seed: int = 12345678,
        get_length_func: callable = None,
        reduce_func: callable = None,
    ) -> None:
        get_length_func = get_length_func or self.get_length
        timestamps = get_length_func(dataset)

        self.timestamps = timestamps
        self.max_timestamp = max_timestamp
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.reduce_func = reduce_func or self._default_reduce_func

    @staticmethod
    def get_length(dataset):
        import torchaudio

        torchaudio.set_audio_backend("sox_io")

        lengths = []
        with dataset.output_keys_as(["wav_path"]):
            for data_index, item in enumerate(
                tqdm(dataset, desc="Read wav_path audio length")
            ):
                info = torchaudio.info(item["wav_path"])
                lengths.append(info.num_frames)
        return lengths

    @staticmethod
    def _default_reduce_func(timestamps):
        return max(timestamps) * len(timestamps)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _evaluate_reduced_timestamps(self, batch_indices):
        return self.reduce_func([self.timestamps[indice] for indice in batch_indices])

    def __iter__(self):
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
