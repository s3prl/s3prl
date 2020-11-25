import random

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset

EXAMPLE_WAV_SAMPLE_NUM = 16000 * 10
EXAMPLE_DATASET_SIZE = 10000


class RandomDataset(Dataset):
    def __init__(self, **kwargs):
        self.class_num = 48

    def __getitem__(self, idx):
        return torch.randn(EXAMPLE_WAV_SAMPLE_NUM), random.randint(0, self.class_num - 1)

    def __len__(self):
        return EXAMPLE_DATASET_SIZE

    def collate_fn(self, samples):
        wavs, labels = [], []
        for wav, label in samples:
            wavs.append(wav)
            labels.append(label)
        return wavs, labels
