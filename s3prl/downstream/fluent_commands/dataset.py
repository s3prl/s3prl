import os
import random
from pathlib import Path

import torch
import torchaudio
import numpy as np
import torch.nn as nn
from torch.utils.data.dataset import Dataset

SAMPLE_RATE = 16000
EXAMPLE_WAV_MAX_SEC = 10


class FluentCommandsDataset(Dataset):
    def __init__(self, df, base_path, Sy_intent):
        self.df = df
        self.base_path = base_path
        self.max_length = SAMPLE_RATE * EXAMPLE_WAV_MAX_SEC
        self.Sy_intent = Sy_intent

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        wav_path = os.path.join(self.base_path, self.df.loc[idx].path)
        wav, sr = torchaudio.load(wav_path)

        wav = wav.squeeze(0)

        label = []

        for slot in ["action", "object", "location"]:
            value = self.df.loc[idx][slot]
            label.append(self.Sy_intent[slot][value])

        return wav.numpy(), np.array(label), Path(wav_path).stem

    def collate_fn(self, samples):
        return zip(*samples)
