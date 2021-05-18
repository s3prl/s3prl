# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ pretrain/byol/dataset.py ]
#   Synopsis     [ the dataset that generates the byol pre-training task data ]
#   Author       [ Andy T. Liu (https://github.com/andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import numpy as np
#-------------#
import torch
from torch.nn.utils.rnn import pad_sequence
import torchaudio
#-------------#
from pretrain.mockingjay.dataset import OnlineAcousticDataset as _OnlineAcousticDataset
from pretrain.byol.task import AudioAugmentationModule, generate_byol_data


HALF_BATCHSIZE_TIME = 99999


class OnlineAcousticDataset(_OnlineAcousticDataset):
    
    def __init__(self, extracter, task_config, bucket_size, file_path, sets, 
                 max_timestep=0, libri_root=None, target_level=-25, **kwargs):
        super(OnlineAcousticDataset, self).__init__(extracter, task_config, bucket_size, file_path, sets, 
              max_timestep, libri_root, target_level, **kwargs)
        self.AudioAugmentationModule = AudioAugmentationModule()

    def _process_x_pad_batch(self, x_pad_batch):
        if self.libri_root is not None:
            x_pad_batch = x_pad_batch.unsqueeze(1) # (batch_size, channel=1, seq_len)
            feat_list = self.extracter(x_pad_batch)
            feat = feat_list[0][:, :self.task_config['sequence_length'], :]
        return generate_byol_data(feat, self.AudioAugmentationModule)

    def collate_fn(self, items):
        items = items[0] # hack bucketing
        return items # __getitem__ should return `spec_stacked`