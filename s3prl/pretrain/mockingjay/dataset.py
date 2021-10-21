# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ dataset.py ]
#   Synopsis     [ the general acoustic dataset, and the datasets that apply the masked reconstruction task ]
#   Author       [ Andy T. Liu (https://github.com/andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import random
#-------------#
import numpy as np
import pandas as pd
#-------------#
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset
import torchaudio
#-------------#
from s3prl.pretrain.mockingjay.task import generate_masked_acoustic_model_data


HALF_BATCHSIZE_TIME = 99999


####################
# ACOUSTIC DATASET #
####################
class AcousticDataset(Dataset):
    
    def __init__(self, extracter, task_config, bucket_size, file_path, sets, 
                 max_timestep=0, libri_root=None, **kwargs):
        super(AcousticDataset, self).__init__()

        self.extracter = extracter
        self.task_config = task_config
        self.libri_root = libri_root
        self.sample_length = task_config['sequence_length'] 
        if self.sample_length > 0:
            print('[Dataset] - Sampling random segments for training, sample length:', self.sample_length)
        
        # Read file
        self.root = file_path
        tables = [pd.read_csv(os.path.join(file_path, s + '.csv')) for s in sets]
        self.table = pd.concat(tables, ignore_index=True).sort_values(by=['length'], ascending=False)
        print('[Dataset] - Training data from these sets:', str(sets))

        # Drop seqs that are too long
        if max_timestep > 0:
            self.table = self.table[self.table.length < max_timestep]
        # Drop seqs that are too short
        if max_timestep < 0:
            self.table = self.table[self.table.length > (-1 * max_timestep)]

        X = self.table['file_path'].tolist()
        X_lens = self.table['length'].tolist()
        self.num_samples = len(X)
        print('[Dataset] - Number of individual training instances:', self.num_samples)

        # Use bucketing to allow different batch size at run time
        self.X = []
        batch_x, batch_len = [], []

        for x, x_len in zip(X, X_lens):
            batch_x.append(x)
            batch_len.append(x_len)
            
            # Fill in batch_x until batch is full
            if len(batch_x) == bucket_size:
                # Half the batch size if seq too long
                if (bucket_size >= 2) and (max(batch_len) > HALF_BATCHSIZE_TIME) and self.sample_length == 0:
                    self.X.append(batch_x[:bucket_size//2])
                    self.X.append(batch_x[bucket_size//2:])
                else:
                    self.X.append(batch_x)
                batch_x, batch_len = [], []
        
        # Gather the last batch
        if len(batch_x) > 1: 
            self.X.append(batch_x)

    def _sample(self, x):
        if self.sample_length <= 0: return x
        if len(x) < self.sample_length: return x
        idx = random.randint(0, len(x)-self.sample_length)
        return x[idx:idx+self.sample_length]

    def __len__(self):
        return len(self.X)

    def collate_fn(self, items):
        items = items[0] # hack bucketing
        assert(len(items) == 5), '__getitem__ should return (spec_masked, pos_enc, mask_label, attn_mask, spec_stacked)'
        return items


class KaldiAcousticDataset(AcousticDataset):
    
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


class OnlineAcousticDataset(AcousticDataset):
    
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