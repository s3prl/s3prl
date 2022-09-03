from collections import defaultdict


class GroupSameItemSampler(object):
    def __init__(
        self,
        dataset,
        item_name: str,
        item_order_name: str,
    ) -> None:
        indices = defaultdict(list)
        with dataset.output_keys_as([item_name, item_order_name]):
            for idx, data_point in enumerate(dataset):
                item = data_point[item_name]
                order = int(data_point[item_order_name])
                indices[item].append((idx, order))

        self.sorted_indices = []
        for item, values in indices.items():
            values.sort(key=lambda x: x[1])
            batch_indices = [idx for idx, order in values]
            self.sorted_indices.append(batch_indices)

        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        for batch_indices in self.sorted_indices:
            yield batch_indices

    def __len__(self):
        return len(list(iter(self)))
