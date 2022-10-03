import logging
import random
from collections import OrderedDict

from s3prl.dataio.sampler import SortedBucketingSampler, SortedSliceSampler
from s3prl.dataset.base import AugmentedDynamicItemDataset

logger = logging.getLogger(__name__)


def get_length(dataset):
    lengths = []
    for index, item in enumerate(dataset):
        lengths.append(item["length"])
    return lengths


def test_sorted_slice_sampler():
    batch_size = 16
    max_length = 16000 * 5
    data = OrderedDict(
        {
            index: {"length": random.randint(16000 * 3, 16000 * 8)}
            for index in range(1000)
        }
    )

    dataset = AugmentedDynamicItemDataset(data)
    sampler = SortedSliceSampler(
        dataset,
        batch_size=batch_size,
        max_length=max_length,
        get_length_func=get_length,
    )

    for epoch in range(5):
        sampler.set_epoch(epoch)
        id2length = get_length(dataset)
        for batch_ids in sampler:
            batch_lengths = [id2length[idx] for idx in batch_ids]
            assert sorted(batch_lengths, reverse=True) == batch_lengths
            if batch_lengths[0] > max_length:
                assert len(batch_lengths) == batch_size // 2

        other_batch_sizes = [
            len(batch)
            for batch in sampler
            if len(batch) not in [batch_size, batch_size // 2]
        ]
        assert len(set(other_batch_sizes)) == len(other_batch_sizes)
        assert len(sampler) == len(data)


def test_sorted_bucketing_sampler():
    batch_size = 16
    max_length = 16000 * 5
    data = OrderedDict(
        {
            index: {"length": random.randint(16000 * 3, 16000 * 8)}
            for index in range(1000)
        }
    )

    dataset = AugmentedDynamicItemDataset(data)
    sampler = SortedBucketingSampler(
        dataset,
        batch_size=batch_size,
        max_length=max_length,
        get_length_func=get_length,
        shuffle=False,
    )

    for epoch in range(5):
        sampler.set_epoch(epoch)
        id2length = get_length(dataset)
        for batch_ids in sampler:
            batch_lengths = [id2length[idx] for idx in batch_ids]
            assert sorted(batch_lengths, reverse=True) == batch_lengths
            if batch_lengths[0] > max_length:
                assert len(batch_lengths) == batch_size // 2

        batch_sizes = [len(batch_indices) for batch_indices in sampler]
        other_batch_sizes = [
            batch_size
            for batch_size in batch_sizes
            if batch_size not in [batch_size, batch_size // 2]
        ]
        assert len(other_batch_sizes) <= 1
        assert len(data) / 16 < len(sampler) < len(data) / 8
