import random

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset

import os
import glob
import torchaudio

class Phonetic(Dataset):
    def __init__(self, split, portion, data_dir):
        self.split = split
        self.portion = portion
        self.data_dir = data_dir
        self.data = glob.glob(os.path.join(self.data_dir, 'phonetic', split+'-'+portion,'*.wav'))

    def __getitem__(self, idx):
        wav, sr = torchaudio.load(self.data[idx])
        return wav.view(-1), self.data[idx]

    def __len__(self):
        return len(self.data)

    def collate_fn(self, samples):
        wavs, paths = [], []
        for wav, path in samples:
            wavs.append(wav)
            paths.append(path)
        return wavs, paths

class Lexical(Dataset):
    pass

class Semantic(Dataset):
    def __init__(self, split, portion, data_dir):
        self.split = split
        self.portion = portion
        self.data_dir = data_dir
        self.data = glob.glob(os.path.join(self.data_dir, 'semantic', split, portion, '*.wav'))

    def __getitem__(self, idx):
        wav, sr = torchaudio.load(self.data[idx])
        return wav.view(-1), self.data[idx]

    def __len__(self):
        return len(self.data)

    def collate_fn(self, samples):
        wavs, paths = [], []
        for wav, path in samples:
            wavs.append(wav)
            paths.append(path)
        return wavs, paths

class Syntactic(Dataset):
    pass

