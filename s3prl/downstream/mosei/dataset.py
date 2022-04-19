import random

import torch
from torch.utils.data.dataset import Dataset

import os
import torchaudio


class MOSEIDataset(Dataset):
    def __init__(self, split, data, path):
        self.split = split
        self.data = data
        self.path = path

    def __getitem__(self, idx):
        wav_path = os.path.join(self.path,
            'Segmented_Audio', self.split, self.data[idx][0])
        wav, sr = torchaudio.load(wav_path)
        label = self.data[idx][1]
        return wav.view(-1), torch.tensor(label).long()

    def __len__(self):
        return len(self.data)

    def collate_fn(self, samples):
        wavs, labels = [], []
        for wav, label in samples:
            wavs.append(wav)
            labels.append(label)
        return wavs, labels
