import copy
from dataclasses import dataclass

import torch

from .base import AugmentedDynamicItemDataset, DataPipe


@dataclass
class AutoregressivePrediction(DataPipe):
    n_future: int = 5
    source_feat_name: str = (
        "source_feat"  # tensors in the shape of: (seq_len, feat_dim)
    )
    target_feat_name: str = (
        "target_feat"  # tensors in the shape of: (seq_len, feat_dim)
    )
    source_feat_len_name: str = "feat_len"

    def generate_shifted_data(self, source_feat):

        with torch.no_grad():

            feat_len = int(source_feat.size(0)) - self.n_future
            target_feat = copy.deepcopy(source_feat[self.n_future :, :])
            source_feat = source_feat[: -self.n_future, :]

            target_feat = target_feat.to(dtype=torch.float32)
            source_feat = source_feat.to(dtype=torch.float32)

        return source_feat, target_feat, feat_len

    def __call__(self, dataset: AugmentedDynamicItemDataset):

        dataset.add_dynamic_item(
            self.generate_shifted_data,
            takes=self.source_feat_name,
            provides=[
                self.source_feat_name,
                self.target_feat_name,
                self.source_feat_len_name,
            ],
        )
        return dataset
