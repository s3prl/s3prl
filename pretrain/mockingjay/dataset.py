# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ dataset.py ]
#   Synopsis     [ the acoustic dataset that will apply the designed pre-training task ]
#   Author       [ S3PRL ]
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
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset
import torchaudio
#-------------#
from pretrain.mockingjay.task import generate_masked_acoustic_model_data


HALF_BATCHSIZE_TIME = 99999


####################
# ACOUSTIC DATASET #
####################
class AcousticDataset(Dataset):
    
    def __init__(self, extracter, task_config, bucket_size, file_path, sets, 
                 max_timestep=0, drop=False, libri_root=None, **kwargs):
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

        # Crop seqs that are too long
        if drop and max_timestep > 0:
            self.table = self.table[self.table.length < max_timestep]

        X = self.table['file_path'].tolist()
        X_lens = self.table['length'].tolist()
        print('[Dataset] - Number of individual training instances:', len(X))

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

    def _get_full_libri_path(self, npy_path):
        # remove .npy
        path = ''.join(npy_path.split('.')[:-1])
        subfolder, filename = path.split('/')
        filedirs = filename.split('-')
        libri_path = os.path.join(self.libri_root, subfolder, filedirs[0], filedirs[1], f'{filename}.flac')
        return libri_path

    def _load_feat(self, npy_path):
        if self.libri_root is None:
            return torch.FloatTensor(np.load(os.path.join(self.root, npy_path)))
        else:
            wav, _ = torchaudio.load(self._get_full_libri_path(npy_path))
            feat = self.extracter(wav.squeeze())
            return feat

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # Load acoustic feature and pad
        x_batch = [self._sample(self._load_feat(x_file)) for x_file in self.X[index]]
        x_pad_batch = pad_sequence(x_batch, batch_first=True)
        return generate_masked_acoustic_model_data(spec=x_pad_batch, config=self.task_config)

    def collate_fn(self, items):
        items = items[0] # hack bucketing
        assert(len(items) == 5), '__getitem__ should return (spec_masked, pos_enc, mask_label, attn_mask, spec_stacked)'
        return items