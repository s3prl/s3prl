# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ pretrain/byol/task.py ]
#   Synopsis     [ Audio Augmentation data processing for pre-training the transformer model ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import torch
from torch import nn
from pretrain.byol.audio_augmentation import RandomResizeCrop, MixupBYOLA, RunningNorm, NormalizeBatch


class AudioAugmentationModule:
    """audio augmentation module, the same parameter with the BYOL-A paper."""

    def __init__(self, log_mixup_exp=True, mixup_ratio=0.4):
        self.train_transform = nn.Sequential(
            MixupBYOLA(ratio=mixup_ratio, log_mixup_exp=log_mixup_exp),
            RandomResizeCrop(virtual_crop_scale=(1.0, 1.0), freq_scale=(0.6, 1.5), time_scale=(0.6, 1.5)),
        )
        self.post_norm = NormalizeBatch()
        print('[Audio Augmentation Module] - Applied Augmentatoions:', self.train_transform)

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)

    def prep_training_step(self, paired_inputs):
        batch_size = paired_inputs[0].shape[0]
        paired_inputs = torch.cat(paired_inputs) # shape: [(B, 1, T, F), (B, 1, T, F)] -> (2*B, 1, T, F)
        paired_inputs = self.post_norm(paired_inputs).squeeze() # shape: (2*B, 1, T, F) -> (2*B, T, F)
        return paired_inputs[:batch_size], paired_inputs[batch_size:] # # shape: (B, T, F)


def generate_byol_data(spec, AudioAugmentationModule):
    """Process training data for byol learning"""
    spec: (B, T, F)
    with torch.no_grad():
        batch_size = spec.shape[0]
        view_1_batch, view_2_batch = [], []
        for idx in range(batch_size):
            view_1, view_2 = AudioAugmentationModule(spec[idx].permute(1, 0).unsqueeze(0)) # shape: (T, F) -> (1, F, T)
            view_1_batch.append(view_1.permute(0, 2, 1)) # shape: (1, F, T) -> (1, T, F)
            view_2_batch.append(view_2.permute(0, 2, 1)) # shape: (1, F, T) -> (1, T, F)
        view_1_batch, view_2_batch = torch.stack(view_1_batch), torch.stack(view_2_batch) # shape: (B, 1, T, F)
        view_1_batch, view_2_batch = AudioAugmentationModule.prep_training_step([view_1_batch, view_2_batch])
    return spec, view_1_batch, view_2_batch # shape: (B, T, F)