from collections import Counter

from s3prl.sampler import BalancedWeightedSampler
from s3prl.dataset.base import AugmentedDynamicItemDataset

def test_balanced_weighted_sampler():
    dataset = AugmentedDynamicItemDataset({
        0: dict(
            label="a"
        ),
        1: dict(
            label="a"
        ),
        2: dict(
            label="b"
        ),
        3: dict(
            label="a"
        ),
    })
    batch_size = 5
    sampler = BalancedWeightedSampler(dataset, batch_size=batch_size, duplicate=1000, seed=0)
    indices = list(sampler)
    assert len(indices[0]) == batch_size

    counter = Counter()
    for batch_indices in indices:
        for idx in batch_indices:
            counter.update(dataset[idx]["label"])

    diff = abs(counter["a"] - counter["b"])
    assert diff < 50
