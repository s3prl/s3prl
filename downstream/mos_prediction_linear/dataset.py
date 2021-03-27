from pathlib import Path

import random
import torch
from torch.utils.data.dataset import Dataset
from torchaudio.sox_effects import apply_effects_file
from itertools import accumulate


class VCC18Dataset(Dataset):
    def __init__(self, dataframe, base_path):
        self.wav_dir = Path(base_path) / "Converted_speech_of_submitted_systems"
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        wav_name, score = self.dataframe.loc[idx]
        wav_path = self.wav_dir / wav_name
        wav, _ = apply_effects_file(
            wav_path,
            [
                ["channels", "1"],
                ["rate", "16000"],
                ["norm"],
            ],
        )

        wav = wav.squeeze(0)
        system_name = wav_name[:3] + wav_name[-8:-4]

        return wav, torch.FloatTensor([score]), system_name

    def collate_fn(self, samples):
        wavs, scores, system_names = zip(*samples)
        return wavs, torch.stack(scores), system_names


class VCC16Dataset(Dataset):
    def __init__(self, wav_list, base_path):
        self.wav_dir = Path(base_path)
        self.wav_list = wav_list

    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, idx):
        wav_name = self.wav_list[idx]
        wav_path = self.wav_dir / wav_name
        wav, _ = apply_effects_file(
            wav_path,
            [
                ["channels", "1"],
                ["rate", "16000"],
                ["norm"],
            ],
        )

        wav = wav.squeeze(0)
        system_name = wav_name.name.split("_")[0]

        return wav, system_name

    def collate_fn(self, samples):
        wavs, system_names = zip(*samples)

        return (
            wavs,
            None,
            system_names,
        )
