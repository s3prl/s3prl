import random

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset

SAMPLE_RATE = 16000
EXAMPLE_WAV_MIN_SEC = 5
EXAMPLE_WAV_MAX_SEC = 15
EXAMPLE_DATASET_SIZE = 1000


class RandomDataset(Dataset):
    def __init__(self, **kwargs):
        self.class_num = 48

    def __getitem__(self, idx):
        wav_sec = random.randint(EXAMPLE_WAV_MIN_SEC, EXAMPLE_WAV_MAX_SEC)
        wav = torch.randn(SAMPLE_RATE * wav_sec)
        label = random.randint(0, self.class_num - 1)
        return wav, label

    def __len__(self):
        return EXAMPLE_DATASET_SIZE

    def collate_fn(self, samples):
        wavs, labels = [], []
        for wav, label in samples:
            wavs.append(wav)
            labels.append(label)
        return wavs, labels
