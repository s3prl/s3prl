from copy import deepcopy
import random
from dataclasses import dataclass
from collections import defaultdict
from .base import AugmentedDynamicItemDataset, DataPipe


@dataclass
class RatioSampler(DataPipe):
    ratio: float = 1.0
    seed: int = 0

    def forward(
        self, dataset: AugmentedDynamicItemDataset
    ) -> AugmentedDynamicItemDataset:
        assert self.ratio <= 1 and self.ratio > 0
        if self.ratio < 1:
            access_ids = list(range(len(dataset)))
            random.seed(self.seed)
            random.shuffle(access_ids)
            target_len = round(len(dataset) * self.ratio)
            new_data_ids = []
            for item_id in access_ids[:target_len]:
                item = dataset[item_id]
                new_data_ids.append(item["id"])

            dataset.data_ids = new_data_ids
        return dataset


@dataclass
class BalancedRatioSampler(DataPipe):
    target_name: str = None
    ratio: float = 1.0
    seed: int = 0

    def forward(
        self, dataset: AugmentedDynamicItemDataset
    ) -> AugmentedDynamicItemDataset:
        assert self.ratio <= 1 and self.ratio > 0
        if self.ratio < 1:
            with dataset.output_keys_as([self.target_name, "id"]):
                access_ids = list(range(len(dataset)))
                label2ids = defaultdict(list)
                random.seed(self.seed)
                random.shuffle(access_ids)
                for item_id in access_ids:
                    item = dataset[item_id]
                    label2ids[item[self.target_name]].append(item["id"])

            target_len = round(len(dataset) * self.ratio)
            new_data_ids = []
            while len(new_data_ids) < target_len:
                for key in sorted(list(label2ids.keys())):
                    ids = label2ids[key]
                    if len(ids) > 0:
                        new_data_ids.append(ids.pop())
                        if len(new_data_ids) == target_len:
                            break

            dataset.data_ids = new_data_ids
        return dataset
