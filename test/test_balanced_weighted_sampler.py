from collections import Counter, OrderedDict

from s3prl.dataset.base import AugmentedDynamicItemDataset
from s3prl.dataio.sampler import BalancedWeightedSampler


def test_balanced_weighted_sampler():
    dataset = AugmentedDynamicItemDataset(
        OrderedDict(
            {
                0: dict(label="a"),
                1: dict(label="a"),
                2: dict(label="b"),
                3: dict(label="a"),
            }
        )
    )
    batch_size = 5
    prev_diff_ratio = 1.0
    for duplicate in [10000, 100000]:
        sampler = BalancedWeightedSampler(
            dataset, batch_size=batch_size, duplicate=duplicate, seed=0
        )
        indices = list(sampler)
        assert len(indices[0]) == batch_size

        counter = Counter()
        for batch_indices in indices:
            for idx in batch_indices:
                counter.update(dataset[idx]["label"])

        diff_ratio = abs(counter["a"] - counter["b"]) / duplicate * len(dataset)
        assert diff_ratio < prev_diff_ratio
        prev_diff_ratio = diff_ratio

    diff_ratio < 0.05
