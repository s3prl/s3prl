import random

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import os 

import csv
import torchaudio

class STDataset(Dataset):
    def __init__(self, clip_dir, tsv_file, tokenizer, max_length = -1):
        super().__init__()
        self.clip_dir = clip_dir
        self.data = []
        self.tokenizer = tokenizer
        with open(tsv_file, 'r') as f:
            for line in csv.DictReader(f, delimiter='\t'):
                self.data.append((line['path'], line['translation']))
        self.max_length = max_length

    def __getitem__(self, idx):
        wav = self._load_wav(self.data[idx][0])
        label = torch.LongTensor(self.tokenizer.encode(self.data[idx][1]))
        if self.max_length > 0:
            label = label[:self.max_length]
        return wav, label

    def __len__(self):
        return len(self.data)

    def collate_fn(self, samples):
        wavs, labels = [], []
        for wav, label in samples:
            wavs.append(wav)
            labels.append(label)
        return wavs, labels

    def _load_wav(self, path):
        wav = torchaudio.load(os.path.join(self.clip_dir, path))[0]
        return wav.view(-1)