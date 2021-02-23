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
TEST_SPEAKERS = [
'mdab0', 'mwbt0', 'felc0', 'mtas1', 'mwew0', 'fpas0',
'mjmp0', 'mlnt0', 'fpkt0', 'mlll0', 'mtls0', 'fjlm0',
'mbpm0', 'mklt0', 'fnlp0', 'mcmj0', 'mjdh0', 'fmgd0',
'mgrt0', 'mnjm0', 'fdhc0', 'mjln0', 'mpam0', 'fmld0']
# Core test set from timit/readme.doc
# Reference1: https://github.com/awni/speech/issues/22
# Reference2: https://github.com/awni/speech/tree/master/examples/timit


#################
# Phone Dataset #
#################
class PhoneDataset(Dataset):
    
    def __init__(self, split, bucket_size, data_root, phone_path, bucket_file, sample_rate=16000, train_dev_seed=1337, **kwargs):
        super(PhoneDataset, self).__init__()
        
        self.data_root = data_root
        self.phone_path = phone_path
        self.sample_rate = sample_rate
        self.class_num = 39 # NOTE: pre-computed, should not need change

        self.Y = {}
        phone_file = open(os.path.join(phone_path, 'converted_aligned_phones.txt')).readlines()
        for line in phone_file:
            line = line.strip('\n').split(' ')
            self.Y[line[0]] = [int(p) for p in line[1:]]
        
        if split == 'train':
            train_list = open(os.path.join(phone_path, 'train_split.txt')).readlines()
            usage_list = [line for line in train_list if line.split('-')[2][:2] in ('SI', 'SX')] # 462 speakers, 3696 sentences, 3.14 hr
        elif split == 'dev' or split == 'test':
            test_list = open(os.path.join(phone_path, 'test_split.txt')).readlines()
            usage_list = [line for line in test_list if line.split('-')[2][:2] != 'SA'] # Standard practice is to remove all "sa" sentences
            if split == 'dev':
                usage_list = [line for line in usage_list if not line.split('-')[1].lower() in TEST_SPEAKERS] # held-out speakers from test
            else:
                usage_list = [line for line in usage_list if line.split('-')[1].lower() in TEST_SPEAKERS] # 24 core test speakers, 192 sentences, 0.16 hr
        else:
            raise ValueError('Invalid \'split\' argument for dataset: PhoneDataset!')
        usage_list = {line.strip('\n'):None for line in usage_list}
        print('[Dataset] - # phone classes: ' + str(self.class_num) + ', number of data for ' + split + ': ' + str(len(usage_list)))

        # Read table for bucketing
        assert os.path.isdir(bucket_file), 'Please first run `preprocess/generate_len_for_bucket.py to get bucket file.'
        table = pd.read_csv(os.path.join(bucket_file, 'TRAIN.csv' if split == 'train' else 'TEST.csv')).sort_values(by=['length'], ascending=False)
        X = table['file_path'].tolist()
        X_lens = table['length'].tolist()

        # Use bucketing to allow different batch sizes at run time
        self.X = []
        batch_x, batch_len = [], []

        for x, x_len in zip(X, X_lens):
            if self._parse_x_name(x).upper() in usage_list:
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
        return '-'.join(x.split('.')[0].split('/')[1:])

    def _load_wav(self, wav_path):
        wav, sr = torchaudio.load(os.path.join(self.data_root, wav_path))
        assert sr == self.sample_rate, f'Sample rate mismatch: real {sr}, config {self.sample_rate}'
        return wav.view(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # Load acoustic feature and pad
        wav_batch = [self._load_wav(x_file) for x_file in self.X[index]]
        label_batch = [torch.LongTensor(self.Y[self._parse_x_name(x_file).upper()]) for x_file in self.X[index]]
        return wav_batch, label_batch # bucketing, return ((wavs, labels))

    def collate_fn(self, items):
        return items[0][0], items[0][1] # hack bucketing, return (wavs, labels)
