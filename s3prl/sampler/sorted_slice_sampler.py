import torch
from collections import OrderedDict
from typing import Iterator, TypeVar
from pydantic import DataclassTypeError

from speechbrain.dataio.sampler import ReproducibleRandomSampler
from torch.utils.data import BatchSampler, RandomSampler, Sampler, SequentialSampler

from .base import Sampler

T_co = TypeVar("T_co", covariant=True)


class SortedSliceSampler(Sampler):
    def __init__(
        self,
        dataset,
        batch_size: int,
        max_length: int = 300000,
        get_length_func: callable = None,
        seed: int = 12345678,
    ) -> None:
        super().__init__(dataset)
        self.epoch = 0
        self.seed = seed
        self.batch_size = batch_size
        self.max_length = max_length

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
            for data_index, item in enumerate(dataset):
                info = torchaudio.info(item["wav_path"])
                length = info.num_frames
                lengths[data_index] = length
        return lengths

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self) -> Iterator[T_co]:
        generator = torch.Generator()
        generator.manual_seed(self.epoch + self.seed)

        indices = torch.randperm(len(self.id2length), generator=generator).tolist()

        batch_size = self.batch_size
        for indice in indices:
            length = self.id2length[indice]
            if length > self.max_length:
                batch_size = self.batch_size // 2
            start_position = self.sorted_ids.index(indice)
            batch = self.sorted_ids[start_position : start_position + batch_size]
            yield batch

    def __len__(self):
        return len(list(iter(self)))
