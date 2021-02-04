# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ dataset.py ]
#   Synopsis     [ the libri speaker dataset ]
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
SPEAKER_THRESHOLD = 0


###################
# Speaker Dataset #
###################
class SpeakerDataset(Dataset):
    
    def __init__(self, split, bucket_size, libri_root, split_file, bucket_file, sample_rate=16000, train_dev_seed=1337, **kwargs):        
        
        self.libri_root = libri_root
        self.split_file = split_file
        self.sample_rate = sample_rate

        # Read table for bucketing
        assert os.path.isdir(bucket_file), 'Please first run `preprocess/generate_len_for_bucket.py to get bucket file.'
        self.table = pd.read_csv(os.path.join(bucket_file, 'train-clean-100.csv')).sort_values(by=['length'], ascending=False)
        X = self.table['file_path'].tolist()
        X_lens = self.table['length'].tolist()
        
        if (split == 'train' or split == 'dev') and os.path.isfile(os.path.join(split_file, 'train_split.txt')):
            usage_list = open(os.path.join(split_file, 'train_split.txt')).readlines()
            random.seed(train_dev_seed)
            random.shuffle(usage_list)
            percent = int(len(usage_list)*0.9)
            usage_list = usage_list[:percent] if split == 'train' else usage_list[percent:]
        elif split == 'test' and os.path.isfile(os.path.join(split_file, 'test_split.txt')):
            usage_list = open(os.path.join(split_file, 'test_split.txt')).readlines()
        else:
            raise NotImplementedError('Invalid `split` argument!')
        usage_list = {line.strip('\n'):None for line in usage_list}

        # Use bucketing to allow different batch sizes at run time
        self.X = []
        batch_x, batch_len = [], []

        for x, x_len in zip(X, X_lens):
            if self._parse_x_name(x) in usage_list: # check if x is in list
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
            if self._parse_x_name(x) in usage_list: # check if x is in list if list not empty
                self.X.append(batch_x)

        # Compute speaker dictionary
        print('[Dataset] - Computing speaker class...')
        speakers = self._get_all_speakers(X)
        self.speaker2idx = self._compute_speaker2idx(speakers)
        self.class_num = len(self.speaker2idx)
        print('[Dataset] - # possible speaker classes: ' + str(self.class_num) + ', number of data for ' + split + ': ' + str(len(usage_list)))

    def _parse_x_name(self, x):
        return x.split('/')[-1].split('.')[0]

    def _load_wav(self, wav_path):
        wav, sr = torchaudio.load(os.path.join(self.libri_root, wav_path))
        # assert sr == self.sample_rate, f'Sample rate mismatch: real {sr}, config {self.sample_rate}'
        return wav.view(-1)

    def _get_speaker_from_path(self, x):
        return x.split('/')[-1].split('.')[0].split('-')[0]

    def _get_all_speakers(self, X):
        speaker_set = {}
        for x in X:
            speaker = self._get_speaker_from_path(x)
            if speaker not in speaker_set:
                speaker_set[speaker] = 0
            else:
                speaker_set[speaker] += 1
        return speaker_set

    def _compute_speaker2idx(self, speakers):
        idx = 0
        speaker2idx = {}
        for speaker in sorted(speakers):
            if speaker not in speaker2idx and speakers[speaker] > SPEAKER_THRESHOLD: # eliminate the speakers with too few utterance
                speaker2idx[speaker] = idx
                idx += 1
        return speaker2idx

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # Load acoustic feature and pad
        wav_batch = [self._load_wav(x_file) for x_file in self.X[index]]
        label_batch = torch.LongTensor([self.speaker2idx[self._get_speaker_from_path(x_file)] for x_file in self.X[index]])
        return wav_batch, label_batch # bucketing, return ((wavs, labels))

    def collate_fn(self, items):
        return items[0][0], items[0][1] # hack bucketing, return (wavs, labels)
