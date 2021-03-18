from pathlib import Path

import torch
from torch.utils.data.dataset import Dataset
from torchaudio.sox_effects import apply_effects_file


class VCC18Dataset(Dataset):
    def __init__(self, dataframe, base_path):
        self.base_path = Path(base_path)
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        wav_name, score = self.dataframe.loc[idx]
        wav_path = self.base_path / "Converted_speech_of_submitted_systems" / wav_name
        wav, _ = apply_effects_file(
            wav_path,
            [
                ["channels", "1"],
                ["rate", "16000"],
                ["norm"],
            ],
        )
        wav = wav.squeeze(0)
        return wav, score

    def collate_fn(self, samples):
        wavs, scores = zip(*samples)
        return wavs, torch.FloatTensor(scores)
