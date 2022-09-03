from collections import defaultdict


class GroupSameItemSampler:
    def __init__(
        self,
        dataset,
        meta_name: str,
    ) -> None:
        self.indices = defaultdict(list)
        for idx in range(len(dataset)):
            meta = dataset.fetch_meta(idx)
            self.indices[meta[meta_name]].append(idx)

        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        for batch_indices in self.indices.values():
            yield batch_indices

    def __len__(self):
        return len(list(iter(self)))
