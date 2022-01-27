# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ dataset.py ]
#   Synopsis     [ The VCC2020 dataset ]
#   Author       [ Wen-Chin Huang (https://github.com/unilight) ]
#   Copyright    [ Copyright(c), Toda Lab, Nagoya University, Japan ]
"""*********************************************************************************************"""


import os
import random

import librosa
import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset

import torchaudio
from .utils import logmelspectrogram

SRCSPKS = ["SEF1", "SEF2", "SEM1", "SEM2"]
FS = 16000 # Always resample to 16kHz

class VCC2020Dataset(Dataset):
    def __init__(self, split, trgspk, data_root, lists_root, fbank_config, train_dev_seed=1337, **kwargs):
        super(VCC2020Dataset, self).__init__()

        self.trgspk = trgspk
        self.trg_lang = trgspk[1]
        self.fbank_config = fbank_config

        X = []
        if split == 'train' or split == 'dev':
            file_list = open(os.path.join(lists_root, self.trg_lang + "_" + split + '_list.txt')).read().splitlines()
            for number in file_list:
                wav_path = os.path.join(data_root, trgspk, number + ".wav")
                if os.path.isfile(wav_path):
                    X.append(wav_path)
            random.seed(train_dev_seed)
            random.shuffle(X)
        elif split == 'test':
            file_list = open(os.path.join(lists_root, 'eval_list.txt')).read().splitlines()
            X = [os.path.join(data_root, srcspk, number + ".wav") for number in file_list for srcspk in SRCSPKS]
        else:
            raise ValueError('Invalid \'split\' argument for dataset: VCC2020Dataset!')
        print('[Dataset] - number of data for ' + split + ': ' + str(len(X)))
        self.X = X

    def _load_wav(self, wav_path, fs):
        # use librosa to resample. librosa gives range [-1, 1]
        wav, sr = librosa.load(wav_path, sr=fs)
        return wav, sr

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        wav_path = self.X[index]
        wav_original, fs_original = self._load_wav(wav_path, fs=None)
        wav_resample, fs_resample = self._load_wav(wav_path, fs=FS)

        lmspc = logmelspectrogram(
            x=wav_original,
            fs=fs_original,
            n_mels=self.fbank_config["n_mels"],
            n_fft=self.fbank_config["n_fft"],
            n_shift=self.fbank_config["n_shift"],
            win_length=self.fbank_config["win_length"],
            window=self.fbank_config["window"],
            fmin=self.fbank_config["fmin"],
            fmax=self.fbank_config["fmax"],
        )

        return wav_resample, wav_original, lmspc, wav_path
    
    def collate_fn(self, batch):
        sorted_batch = sorted(batch, key=lambda x: -x[1].shape[0])
        bs = len(sorted_batch) # batch_size
        wavs = [torch.from_numpy(sorted_batch[i][0]) for i in range(bs)]
        wavs_2 = [torch.from_numpy(sorted_batch[i][1]) for i in range(bs)] # This is used for obj eval
        acoustic_features = [torch.from_numpy(sorted_batch[i][2]) for i in range(bs)]
        acoustic_features_padded = pad_sequence(acoustic_features, batch_first=True)
        acoustic_feature_lengths = torch.from_numpy(np.array([acoustic_feature.size(0) for acoustic_feature in acoustic_features]))
        wav_paths = [sorted_batch[i][3] for i in range(bs)]
        
        return wavs, wavs_2, acoustic_features, acoustic_features_padded, acoustic_feature_lengths, wav_paths


class CustomDataset(Dataset):
    def __init__(self, eval_list_file, **kwargs):
        super(CustomDataset, self).__init__()

        X = []
        if os.path.isfile(eval_list_file):
            print("[Dataset] Reading custom eval list file: {}".format(eval_list_file))
            X = open(eval_list_file, "r").read().splitlines()
        else:
            raise ValueError("[Dataset] eval list file does not exist: {}".format(eval_list_file))
        print('[Dataset] - number of data for custom test: ' + str(len(X)))
        self.X = X

    def _load_wav(self, wav_path, fs):
        # use librosa to resample. librosa gives range [-1, 1]
        wav, sr = librosa.load(wav_path, sr=fs)
        return wav, sr

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        wav_path = self.X[index]
        wav_original, fs_original = self._load_wav(wav_path, fs=None)
        wav_resample, fs_resample = self._load_wav(wav_path, fs=FS)

        return wav_resample, wav_original, wav_path
    
    def collate_fn(self, batch):
        sorted_batch = sorted(batch, key=lambda x: -x[1].shape[0])
        bs = len(sorted_batch) # batch_size
        wavs = [torch.from_numpy(sorted_batch[i][0]) for i in range(bs)]
        wavs_2 = [torch.from_numpy(sorted_batch[i][1]) for i in range(bs)] # This is used for obj eval
        wav_paths = [sorted_batch[i][2] for i in range(bs)]
        
        return wavs, wavs_2, None, None, None, wav_paths