import os
import random

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import torchaudio

ORIGINAL_SAMPLE_RATE = 22050
SAMPLE_RATE = 16000
EXAMPLE_WAV_MAX_SEC = 10


class AudioSLUDataset(Dataset):
    def __init__(self, df, base_path, Sy_intent, speaker_name):
        self.df = df
        self.base_path = base_path
        self.max_length = SAMPLE_RATE * EXAMPLE_WAV_MAX_SEC
        self.Sy_intent = Sy_intent
        self.speaker_name = speaker_name
        self.resampler = torchaudio.transforms.Resample(ORIGINAL_SAMPLE_RATE, SAMPLE_RATE)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        wav_path = os.path.join(self.base_path, 'audio_'+self.speaker_name , 'snips',self.df.loc[idx]['u_id']+'.mp3')
        wav, sr = torchaudio.load(wav_path)
        assert sr == ORIGINAL_SAMPLE_RATE
        wav = self.resampler(wav)
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


        

