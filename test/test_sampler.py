import pytest
import logging

import torch

from s3prl import Output
from s3prl.dataset import Dataset
from s3prl.sampler import (
    DistributedBatchSamplerWrapper,
    MaxTimestampBatchSampler,
    FixedBatchSizeBatchSampler,
)

logger = logging.getLogger(__name__)


class TimestampDataset(Dataset):
    def __init__(self, timestamps) -> None:
        super().__init__()
        self.timestamps = timestamps
        self.data = [torch.randn(timestamp) for timestamp in timestamps]

    def __len__(self):
        return len(self.timestamps)

    def __getitem__(self, index):
        if self.in_metadata_mode():
            return Output(timestamp=self.timestamps[index])
        return Output(output=self.data[index])

    def collate_fn(self, samples):
        return zip(*samples)


def _merge_batch_indices(batch_indices):
    all_indices = []
    for indices in batch_indices:
        all_indices += indices
    return all_indices


@pytest.mark.parametrize("world_size", [1, 2, 3, 4, 5, 6, 7, 8])
def test_distributed_sampler(world_size):
    sampler = [[1, 2, 3], [4, 5, 6, 7], [8], [9, 10]]
    ddp_indices = []
    for rank in range(world_size):
        ddp_sampler = DistributedBatchSamplerWrapper(sampler, world_size, rank)
        ddp_indices += _merge_batch_indices(ddp_sampler)
    assert sorted(ddp_indices) == sorted(_merge_batch_indices(sampler))


timestamps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


@pytest.mark.parametrize("max_timestamp", [10, 11, 12, 20, sum(timestamps)])
def test_MaxTimestampBatchSampler(max_timestamp):
    dataset = TimestampDataset(timestamps)
    iter1 = list(iter(MaxTimestampBatchSampler(dataset, max_timestamp, shuffle=False)))
    iter2 = list(iter(MaxTimestampBatchSampler(dataset, max_timestamp, shuffle=True)))
    indices1 = sorted(_merge_batch_indices(iter1))
    indices2 = sorted(_merge_batch_indices(iter2))
    assert indices1 == indices2 == list(range(len(timestamps)))


data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


@pytest.mark.parametrize("batch_size", [1, 2, 3, len(data)])
def test_FixedBatchSizeBatchSampler(batch_size):
    dataset = data
    iter1 = list(iter(FixedBatchSizeBatchSampler(dataset, batch_size, shuffle=False)))
    iter2 = list(iter(FixedBatchSizeBatchSampler(dataset, batch_size, shuffle=True)))
    indices1 = sorted(_merge_batch_indices(iter1))
    indices2 = sorted(_merge_batch_indices(iter2))
    assert indices1 == indices2 == list(range(len(timestamps)))
