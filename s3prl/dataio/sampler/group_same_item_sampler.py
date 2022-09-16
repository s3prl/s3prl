"""
Group the data points with the same key into the same batch

Authors:
  * Shu-wen Yang 2022
"""

from collections import defaultdict

__all__ = [
    "GroupSameItemSampler",
]


class GroupSameItemSampler:
    def __init__(
        self,
        dataset,
        item: str,
    ) -> None:
        self.indices = defaultdict(list)
        for idx in range(len(dataset)):
            if hasattr(dataset, "get_info"):
                info = dataset.get_info(idx)
            else:
                info = dataset[idx]
            self.indices[info[item]].append(idx)

        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        for batch_indices in self.indices.values():
            yield batch_indices

    def __len__(self):
        return len(list(iter(self)))
