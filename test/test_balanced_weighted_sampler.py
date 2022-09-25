from collections import Counter

import pytest

from s3prl.dataio.sampler import BalancedWeightedSampler


@pytest.mark.parametrize("duplicate", [10000, 100000])
def test_balanced_weighted_sampler(duplicate: int):
    labels = ["a", "a", "b", "a"]
    batch_size = 5
    prev_diff_ratio = 1.0
    sampler = BalancedWeightedSampler(
        labels, batch_size=batch_size, duplicate=duplicate, seed=0
    )
    indices = list(sampler)
    assert len(indices[0]) == batch_size

    counter = Counter()
    for batch_indices in indices:
        for idx in batch_indices:
            counter.update(labels[idx])

    diff_ratio = abs(counter["a"] - counter["b"]) / duplicate * len(labels)
    assert diff_ratio < prev_diff_ratio
    prev_diff_ratio = diff_ratio

    diff_ratio < 0.05
