# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ pretrain/apc/dataset.py ]
#   Synopsis     [ the dataset that applies the apc preprocessing on audio ]
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
from pretrain.bucket_dataset import FeatDataset


class ApcAudioDataset(FeatDataset):
    
    def __init__(self, extracter, task_config, bucket_size, file_path, sets, 
                 max_timestep=0, libri_root=None, **kwargs):
        super(ApcAudioDataset, self).__init__(extracter, task_config, bucket_size, file_path, sets, 
                                                   max_timestep, libri_root, **kwargs)

    def _load_feat(self, feat_path):
        if self.libri_root is None:
            return torch.FloatTensor(np.load(os.path.join(self.root, feat_path)))
        else:
            wav, _ = torchaudio.load(os.path.join(self.libri_root, feat_path))
            feat = self.extracter(wav)
            return feat

    def __getitem__(self, index):
        # Load acoustic feature and pad
        x_batch = [self._sample(self._load_feat(x_file)) for x_file in self.X[index]]
        x_len = [len(x_b) for x_b in x_batch]
        x_pad_batch = pad_sequence(x_batch, batch_first=True)
        return x_pad_batch, x_len
