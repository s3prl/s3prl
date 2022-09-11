from collections import defaultdict


class GroupSameItemSampler:
    def __init__(
        self,
        dataset,
        item: str,
    ) -> None:
        self.indices = defaultdict(list)
        for idx in range(len(dataset)):
            info = dataset.get_info(idx)
            self.indices[info[item]].append(idx)

        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        for batch_indices in self.indices.values():
            yield batch_indices

    def __len__(self):
        return len(list(iter(self)))
