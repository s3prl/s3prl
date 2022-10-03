import logging
import random
from collections import OrderedDict

from s3prl.dataio.sampler import SortedBucketingSampler, SortedSliceSampler

logger = logging.getLogger(__name__)


def test_sorted_slice_sampler():
    batch_size = 16
    max_length = 16000 * 5
    lengths = [random.randint(16000 * 3, 16000 * 8) for index in range(1000)]

    sampler = SortedSliceSampler(
        lengths,
        batch_size=batch_size,
        max_length=max_length,
    )

    for epoch in range(5):
        sampler.set_epoch(epoch)
        id2length = lengths
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
        assert len(sampler) == len(lengths)


def test_sorted_bucketing_sampler():
    batch_size = 16
    max_length = 16000 * 5
    lengths = [random.randint(16000 * 3, 16000 * 8) for index in range(1000)]

    sampler = SortedBucketingSampler(
        lengths,
        batch_size=batch_size,
        max_length=max_length,
        shuffle=False,
    )

    for epoch in range(5):
        sampler.set_epoch(epoch)
        id2length = lengths
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
        assert len(lengths) / 16 < len(sampler) < len(lengths) / 8
