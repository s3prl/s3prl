import logging
import random
from s3prl.dataset.base import AugmentedDynamicItemDataset
from s3prl.sampler import SortedSliceSampler

logger = logging.getLogger(__name__)

def get_length(dataset):
    id2length = {}
    for index, item in enumerate(dataset):
        id2length[index] = item["length"]
    return id2length

def test_sorted_slice_sampler():
    max_length = 16000 * 5
    data = {index: {"length": random.randint(16000 * 3, 16000 * 8)} for index in range(1000)}

    dataset = AugmentedDynamicItemDataset(data)
    sampler = SortedSliceSampler(dataset, batch_size=16, max_length=max_length, get_length_func=get_length)

    for epoch in range(5):
        sampler.set_epoch(epoch)
        id2length = get_length(dataset)
        for batch_ids in sampler:
            batch_lengths = [id2length[idx] for idx in batch_ids]
            assert sorted(batch_lengths, reverse=True) == batch_lengths
            if batch_lengths[0] > max_length:
                assert len(batch_lengths) == 8
