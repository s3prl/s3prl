# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ dataset.py ]
#   Synopsis     [ the phone dataset ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import random
#-------------#
import pandas as pd
#-------------#
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset
#-------------#
import torchaudio


HALF_BATCHSIZE_TIME = 2000


#################
# Phone Dataset #
#################
class PhoneDataset(Dataset):
    
    def __init__(self, split, bucket_size, libri_root, phone_path, bucket_file, sample_rate=16000, train_dev_seed=1337, **kwargs):
        super(PhoneDataset, self).__init__()
        
        self.libri_root = libri_root
        self.phone_path = phone_path
        self.sample_rate = sample_rate
        self.class_num = 41 # NOTE: pre-computed, should not need change

        self.Y = {}
        phone_file = open(os.path.join(phone_path, 'converted_aligned_phones.txt')).readlines()
        for line in phone_file:
            line = line.strip('\n').split(' ')
            self.Y[line[0]] = [int(p) for p in line[1:]]
        
        if split == 'train' or split == 'dev':
            usage_list = open(os.path.join(phone_path, 'train_split.txt')).readlines()
            random.seed(train_dev_seed)
            random.shuffle(usage_list)
            percent = int(len(usage_list)*0.9)
            usage_list = usage_list[:percent] if split == 'train' else usage_list[percent:]
        elif split == 'test':
            usage_list = open(os.path.join(phone_path, 'test_split.txt')).readlines()
        else:
            raise ValueError('Invalid \'split\' argument for dataset: PhoneDataset!')
        usage_list = {line.strip('\n'):None for line in usage_list}
        print('[Dataset] - # phone classes: ' + str(self.class_num) + ', number of data for ' + split + ': ' + str(len(usage_list)))

        # Read table for bucketing
        assert os.path.isdir(bucket_file), 'Please first run `preprocess/generate_len_for_bucket.py to get bucket file.'
        table = pd.read_csv(os.path.join(bucket_file, 'train-clean-100.csv')).sort_values(by=['length'], ascending=False)
        X = table['file_path'].tolist()
        X_lens = table['length'].tolist()

        # Use bucketing to allow different batch sizes at run time
        self.X = []
        batch_x, batch_len = [], []

        for x, x_len in zip(X, X_lens):
            if self._parse_x_name(x) in usage_list:
                batch_x.append(x)
                batch_len.append(x_len)
                
                # Fill in batch_x until batch is full
                if len(batch_x) == bucket_size:
                    # Half the batch size if seq too long
                    if (bucket_size >= 2) and (max(batch_len) > HALF_BATCHSIZE_TIME):
                        self.X.append(batch_x[:bucket_size//2])
                        self.X.append(batch_x[bucket_size//2:])
                    else:
                        self.X.append(batch_x)
                    batch_x, batch_len = [], []
        
        # Gather the last batch
        if len(batch_x) > 1:
            if self._parse_x_name(x) in usage_list:
                self.X.append(batch_x)

    def _parse_x_name(self, x):
        return x.split('/')[-1].split('.')[0]

    def _load_wav(self, wav_path):
        wav, sr = torchaudio.load(os.path.join(self.libri_root, wav_path))
        # assert sr == self.sample_rate, f'Sample rate mismatch: real {sr}, config {self.sample_rate}'
        return wav.view(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # Load acoustic feature and pad
        wav_batch = [self._load_wav(x_file) for x_file in self.X[index]]
        label_batch = [torch.LongTensor(self.Y[self._parse_x_name(x_file)]) for x_file in self.X[index]]
        return wav_batch, label_batch # bucketing, return ((wavs, labels))

    def collate_fn(self, items):
        return items[0][0], items[0][1] # hack bucketing, return (wavs, labels)
