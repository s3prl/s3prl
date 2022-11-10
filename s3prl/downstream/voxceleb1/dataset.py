import os
import random
import logging
from pathlib import Path

import torchaudio
import pandas as pd
from torch.utils.data import Dataset

from s3prl.dataio.encoder import CategoryEncoder

log = logging.getLogger(__name__)

CACHE_PATH = os.path.join(os.path.dirname(__file__), ".cache/")


# Voxceleb 1 Speaker Identification
class SpeakerClassifiDataset(Dataset):
    def __init__(self, mode: str, csv_dir: str, max_timestep: int = None):
        df = pd.read_csv(Path(csv_dir) / f"{mode}.csv")

        self.wav_paths = df["wav_path"].tolist()
        spks = df["label"].tolist()
        self._encoder = CategoryEncoder(spks)
        self.label = [self.encoder.encode(spk) for spk in spks]
        self.id = df["id"].tolist()

        self._speaker_num = len(set(spks))
        self.max_timestep = max_timestep

    @property
    def encoder(self):
        return self._encoder

    @property
    def speaker_num(self):
        return self._speaker_num

    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, idx):
        path = self.wav_paths[idx]
        wav, sr = torchaudio.load(path)
        wav = wav.squeeze(0)
        length = wav.shape[0]

        if self.max_timestep != None:
            if length > self.max_timestep:
                start = random.randint(0, int(length - self.max_timestep))
                wav = wav[start : start + self.max_timestep]
                length = self.max_timestep

        return wav.numpy(), self.label[idx], self.id[idx]

    def collate_fn(self, samples):
        return zip(*samples)
