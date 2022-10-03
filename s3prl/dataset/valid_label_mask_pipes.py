from dataclasses import dataclass

import torch

from .base import AugmentedDynamicItemDataset, DataPipe


@dataclass
class LabelMaskFromLen(DataPipe):
    target_feat_name: str = (
        "target_feat"  # tensors in the shape of: (seq_len, feat_dim)
    )
    label_mask_name: str = "label_mask"

    def create_label_mask(self, target_feat):

        with torch.no_grad():

            seq_len = target_feat.shape[0]
            label_mask = torch.ones_like(target_feat, dtype=torch.uint8)
            label_mask[seq_len:, :] = 0
            label_mask = label_mask.to(dtype=torch.bool)

        return label_mask

    def __call__(self, dataset: AugmentedDynamicItemDataset):

        dataset.add_dynamic_item(
            self.create_label_mask,
            takes=[self.target_feat_name],
            provides=[
                self.label_mask_name,
            ],
        )
        return dataset
