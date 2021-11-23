# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ dataset.py ]
#   Synopsis     [ the dataset that applies the spec augment pre-training task ]
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
from pretrain.mockingjay.dataset import KaldiAcousticDataset as _KaldiAcousticDataset
from pretrain.mockingjay.dataset import OnlineAcousticDataset as _OnlineAcousticDataset
from pretrain.spec_augment.task import generate_spec_aug_data


HALF_BATCHSIZE_TIME = 99999


class KaldiAcousticDataset(_KaldiAcousticDataset):
    
    def __init__(self, extracter, task_config, bucket_size, file_path, sets, 
                 max_timestep=0, libri_root=None, **kwargs):
        super(KaldiAcousticDataset, self).__init__(extracter, task_config, bucket_size, file_path, sets, 
                                                   max_timestep, libri_root, **kwargs)

    def __getitem__(self, index):
        # Load acoustic feature and pad
        x_batch = [self._sample(self._load_feat(x_file)) for x_file in self.X[index]]
        x_pad_batch = pad_sequence(x_batch, batch_first=True)
        return generate_spec_aug_data(spec=(x_pad_batch,), config=self.task_config)


class OnlineAcousticDataset(_OnlineAcousticDataset):
    
    def __init__(self, extracter, task_config, bucket_size, file_path, sets, 
                 max_timestep=0, libri_root=None, target_level=-25, **kwargs):
        super(OnlineAcousticDataset, self).__init__(extracter, task_config, bucket_size, file_path, sets, 
                                                    max_timestep, libri_root, target_level, **kwargs)
  
    def _process_x_pad_batch(self, x_pad_batch):
        if self.libri_root is not None:
            x_pad_batch = x_pad_batch.unsqueeze(1) # (batch_size, channel=1, seq_len)
            feat_list = self.extracter(x_pad_batch)
        return generate_spec_aug_data(feat_list, config=self.task_config)
