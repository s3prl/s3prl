"""
The most commonly used batch sampler in S3PRL legacy codebase,
which sorts the lengths of all the data points and group the instances
with the similar lengths together.

Authors:
  Shu-wen Yang 2022
"""

import torch
from tqdm import tqdm

__all__ = [
    "SortedSliceSampler",
    "SortedBucketingSampler",
]


class SortedSliceSampler:
    """
    This sampler should only be used for training hence is always in random shuffle mode

    Args:
        dataset (DynamicItemDataset)
        batch_size (int): the default batch size
        max_length (int): if a batch contains at least on utt longer than max_length, half the batch
        get_length_func (callable): get the length of each item in the dataset, if None, a default function will be used
        in_batch_shuffle (bool): if False, batches are sorted by length from long to short
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        max_length: int = 300000,
        get_length_func: callable = None,
        seed: int = 12345678,
        in_batch_shuffle: bool = False,
    ) -> None:
        self.epoch = 0
        self.seed = seed
        self.batch_size = batch_size
        self.max_length = max_length
        self.in_batch_shuffle = in_batch_shuffle

        get_length_func = get_length_func or self.get_length
        self.id2length = get_length_func(dataset)
        sorted_ids = [(idx, length) for idx, length in self.id2length.items()]
        sorted_ids = sorted(sorted_ids, key=lambda x: x[1], reverse=True)
        self.sorted_ids = [data_id for data_id, length in sorted_ids]

    @staticmethod
    def get_length(dataset):
        import torchaudio

        torchaudio.set_audio_backend("sox_io")

        lengths = {}
        with dataset.output_keys_as(["wav_path"]):
            for data_index, item in enumerate(
                tqdm(dataset, desc="Read wav_path audio length")
            ):
                info = torchaudio.info(item["wav_path"])
                length = info.num_frames
                lengths[data_index] = length
        return lengths

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.epoch + self.seed)

        indices = torch.randperm(len(self.id2length), generator=generator).tolist()

        for indice in indices:
            length = self.id2length[indice]
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
        dataset (DynamicItemDataset)
        batch_size (int): the default batch size
        max_length (int): if a batch contains at least on utt longer than max_length, half the batch
        get_length_func (callable): get the length of each item in the dataset, if None, a default function will be used
        shuffle (bool): Whether to shuffle the batches
        in_batch_shuffle (bool): if False, batches are sorted by length from long to short
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        max_length: int = 300000,
        get_length_func: callable = None,
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

        get_length_func = get_length_func or self.get_length
        self.id2length = get_length_func(dataset)
        sorted_ids = [(idx, length) for idx, length in self.id2length.items()]

        # sorted_ids should be from long -> short utts
        sorted_ids = sorted(sorted_ids, key=lambda x: x[1], reverse=True)
        self.sorted_ids = [data_id for data_id, length in sorted_ids]

    @staticmethod
    def get_length(dataset):
        import torchaudio

        torchaudio.set_audio_backend("sox_io")

        lengths = {}
        with dataset.output_keys_as(["wav_path"]):
            for data_index, item in enumerate(
                tqdm(dataset, desc="Read wav_path audio length")
            ):
                info = torchaudio.info(item["wav_path"])
                length = info.num_frames
                lengths[data_index] = length
        return lengths

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.epoch + self.seed)

        batches = []
        position = 0
        while position < len(self.sorted_ids):
            indice = self.sorted_ids[position]
            length = self.id2length[indice]
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
