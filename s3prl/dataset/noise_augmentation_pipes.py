import random
from dataclasses import dataclass

import torch

from .base import AugmentedDynamicItemDataset, DataPipe


@dataclass
class NoiseAugmentation(DataPipe):
    noise_args: dict = None
    input_feat_name: str = "input_feat"  # tensors in the shape of: (seq_len, feat_dim)
    output_feat_name: str = (
        "output_feat"  # tensors in the shape of: (seq_len, feat_dim)
    )

    def apply_noise_on_data(self, input_feat):

        with torch.no_grad():
            if self.noise_args["noise_proportion"] > 0:
                # noise augmentation
                dice = random.random()
                if dice < self.noise_args["noise_proportion"]:
                    noise_sampler = torch.distributions.Normal(0, 0.2)
                    input_feat += noise_sampler.sample(input_feat.shape).to(
                        device=input_feat.device
                    )
            input_feat = input_feat.to(dtype=torch.float32)
        return input_feat

    def __call__(self, dataset: AugmentedDynamicItemDataset):

        dataset.add_dynamic_item(
            self.apply_noise_on_data,
            takes=self.input_feat_name,
            provides=self.output_feat_name,
        )
        return dataset
