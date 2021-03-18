import os

import torch
import torchaudio
from torch.utils.data.dataset import Dataset

SAMPLE_RATE = 16000


class VCC18Dataset(Dataset):
    def __init__(self, dataframe, base_path):
        self.base_path = base_path
        self.dataframe = dataframe
        self.downsample = None
        self.length = len(dataframe)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        wav_path = os.path.join(
            self.base_path,
            "Converted_speech_of_submitted_systems",
            self.dataframe.loc[idx][0],
        )
        wav, ORIG_SAMPLE_RATE = torchaudio.load(wav_path)
        if self.downsample is None:
            self.donwsample = torchaudio.transforms.Resample(
                ORIG_SAMPLE_RATE, SAMPLE_RATE, resampling_method="sinc_interpolation"
            )
        wav = self.donwsample(wav)

        wav = wav.squeeze(0)

        score = self.dataframe.loc[idx][1]

        return wav, score

    def collate_fn(self, samples):
        wavs, scores = zip(*samples)

        return wavs, torch.FloatTensor(scores)
