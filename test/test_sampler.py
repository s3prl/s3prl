import logging

import pytest

from s3prl.dataio.sampler import (
    DistributedBatchSamplerWrapper,
    FixedBatchSizeBatchSampler,
    MaxTimestampBatchSampler,
)

logger = logging.getLogger(__name__)


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
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


@pytest.mark.parametrize("batch_size", [1, 2, 3, len(data)])
def test_FixedBatchSizeBatchSampler(batch_size):
    dataset = data
    iter1 = list(iter(FixedBatchSizeBatchSampler(dataset, batch_size, shuffle=False)))
    iter2 = list(iter(FixedBatchSizeBatchSampler(dataset, batch_size, shuffle=True)))
    indices1 = sorted(_merge_batch_indices(iter1))
    indices2 = sorted(_merge_batch_indices(iter2))
    assert indices1 == indices2 == list(range(len(timestamps)))
