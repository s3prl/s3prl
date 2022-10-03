"""
Group the data points with the same key into the same batch

Authors:
  * Leo 2022
"""

from collections import defaultdict
from typing import List

__all__ = [
    "GroupSameItemSampler",
]


class GroupSameItemSampler:
    def __init__(
        self,
        items: List[str],
    ) -> None:
        self.indices = defaultdict(list)
        for idx, item in enumerate(items):
            self.indices[item].append(idx)

        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        for batch_indices in self.indices.values():
            yield batch_indices

    def __len__(self):
        return len(list(iter(self)))
