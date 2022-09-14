import copy
import random
from dataclasses import dataclass

import torch

from .base import AugmentedDynamicItemDataset, DataPipe


@dataclass
class NoiseAugmentation(DataPipe):
    noise_proportion: float = 0.0
    input_feat_name: str = "input_feat"
    output_feat_name: str = "output_feat"
    """
    Args:
        noise_proportion (float): for this percentage of the time, Gaussian noise will be applied on all frames during MAM training, set to 0 for no noise
        input_feat_name (str): handle for the `takes` (input)
        output_feat_name (str): handle for the `provides` (output)
    """

    def apply_noise_on_data(self, input_feat):

        with torch.no_grad():
            if self.noise_proportion > 0:
                noised_feat = copy.deepcopy(input_feat)
                dice = random.random()
                if dice < self.noise_proportion:
                    noise_sampler = torch.distributions.Normal(0, 0.2)
                    noised_feat += noise_sampler.sample(noised_feat.shape)
                noised_feat = noised_feat.to(dtype=torch.float32)
                return noised_feat
            else:
                return input_feat

    def __call__(self, dataset: AugmentedDynamicItemDataset):

        dataset.add_dynamic_item(
            self.apply_noise_on_data,
            takes=self.input_feat_name,
            provides=self.output_feat_name,
        )
        return dataset
