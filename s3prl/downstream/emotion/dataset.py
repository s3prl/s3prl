# -*- coding: utf-8 -*- #
"""
    FileName     [ dataset.py ]
    Synopsis     [ the emotion classifier dataset ]
    Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""

import logging
from pathlib import Path

import torchaudio
import pandas as pd
from torch.utils.data import Dataset
from torchaudio.transforms import Resample

SAMPLE_RATE = 16000
log = logging.getLogger(__name__)

class IEMOCAPDataset(Dataset):
    def __init__(self, split: str, csv_dir: str, pre_load=True):
        self.pre_load = pre_load

        assert split in ["train", "dev", "test"]
        df = pd.read_csv(Path(csv_dir) / f"{split}.csv")

        self.wav_paths = df["wav_path"].tolist()
        self.labels = df["label"].tolist()

        classes = sorted(set(self.labels))
        self.class_num = len(classes)
        self.idx2emotion = {idx: label for idx, label in enumerate(classes)}
        self.emotion2idx = {label: idx for idx, label in enumerate(classes)}

        _, origin_sr = torchaudio.load(self.wav_paths[0])
        self.resampler = Resample(origin_sr, SAMPLE_RATE)
        if self.pre_load:
            self.wavs = self._load_all()

    def _load_wav(self, path):
        wav, _ = torchaudio.load(path)
        wav = self.resampler(wav).squeeze(0)
        return wav

    def _load_all(self):
        wavforms = []
        for wav_path in self.wav_paths:
            wav = self._load_wav(wav_path)
            wavforms.append(wav)
        return wavforms

    def __getitem__(self, idx):
        wav_path = self.wav_paths[idx]
        label_id = self.emotion2idx[self.labels[idx]]
        if self.pre_load:
            wav = self.wavs[idx]
        else:
            wav = self._load_wav(wav_path)
        return wav.numpy(), label_id, Path(wav_path).stem

    def __len__(self):
        return len(self.wav_paths)

def collate_fn(samples):
    return zip(*samples)
