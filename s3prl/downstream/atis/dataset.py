import random

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset


import os
import torchaudio


SAMPLE_RATE = 16000
EXAMPLE_WAV_MAX_SEC = 10




class AtisDataset(Dataset):
    def __init__(self, df, base_path, Sy_intent, type):
        self.df = df
        self.base_path = base_path
        self.max_length = SAMPLE_RATE * EXAMPLE_WAV_MAX_SEC
        self.Sy_intent = Sy_intent
        self.type = type
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        wav_path = os.path.join(self.base_path, self.type, self.df.loc[idx]['id']+'.wav')
        wav, sr = torchaudio.load(wav_path)
        wav = wav.squeeze(0)
        label = []

        for slot in ["intent"]:
            value = self.df.loc[idx][slot]
            label.append(self.Sy_intent[slot][value])

        return wav, torch.tensor(label).long()

    def collate_fn(self, samples):
        wavs, labels = [], []

        for wav, label in samples:
            wavs.append(wav)
            labels.append(label)

        return wavs, labels

