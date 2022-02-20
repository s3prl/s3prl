# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ pretrain/mockingjay/dataset.py ]
#   Synopsis     [ The bucketing datasets that apply the masked reconstruction task on-the-fly ]
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
from pretrain.mockingjay.task import generate_masked_acoustic_model_data
from pretrain.bucket_dataset import FeatDataset


HALF_BATCHSIZE_TIME = 99999


class KaldiAcousticDataset(FeatDataset):
    
    def __init__(self, extracter, task_config, bucket_size, file_path, sets, 
                 max_timestep=0, libri_root=None, **kwargs):
        super(KaldiAcousticDataset, self).__init__(extracter, task_config, bucket_size, file_path, sets, 
                                                   max_timestep, libri_root, **kwargs)

    def _load_feat(self, feat_path):
        if self.libri_root is None:
            return torch.FloatTensor(np.load(os.path.join(self.root, feat_path)))
        else:
            wav, _ = torchaudio.load(os.path.join(self.libri_root, feat_path))
            feat = self.extracter(wav.squeeze())
            return feat

    def __getitem__(self, index):
        # Load acoustic feature and pad
        x_batch = [self._sample(self._load_feat(x_file)) for x_file in self.X[index]]
        x_pad_batch = pad_sequence(x_batch, batch_first=True)
        return generate_masked_acoustic_model_data(spec=(x_pad_batch,), config=self.task_config)


class OnlineAcousticDataset(FeatDataset):
    
    def __init__(self, extracter, task_config, bucket_size, file_path, sets, 
                 max_timestep=0, libri_root=None, target_level=-25, **kwargs):
        max_timestep *= 160
        super(OnlineAcousticDataset, self).__init__(extracter, task_config, bucket_size, file_path, sets, 
                                                    max_timestep, libri_root, **kwargs)
        self.target_level = target_level
        self.sample_length = self.sample_length * 160
    
    def _normalize_wav_decibel(self, wav):
        '''Normalize the signal to the target level'''
        if self.target_level == 'None':
            return wav
        rms = wav.pow(2).mean().pow(0.5)
        scalar = (10 ** (self.target_level / 20)) / (rms + 1e-10)
        wav = wav * scalar
        return wav

    def _load_feat(self, feat_path):
        if self.libri_root is None:
            return torch.FloatTensor(np.load(os.path.join(self.root, feat_path)))
        else:
            wav, _ = torchaudio.load(os.path.join(self.libri_root, feat_path))
            wav = self._normalize_wav_decibel(wav.squeeze())
            return wav # (seq_len)

    def _process_x_pad_batch(self, x_pad_batch):
        if self.libri_root is not None:
            x_pad_batch = x_pad_batch.unsqueeze(1) # (batch_size, channel=1, seq_len)
            feat_list = self.extracter(x_pad_batch)
        return generate_masked_acoustic_model_data(feat_list, config=self.task_config)

    def __getitem__(self, index):
        # Load acoustic feature and pad
        x_batch = [self._sample(self._load_feat(x_file)) for x_file in self.X[index]]
        x_pad_batch = pad_sequence(x_batch, batch_first=True)
        return self._process_x_pad_batch(x_pad_batch)
