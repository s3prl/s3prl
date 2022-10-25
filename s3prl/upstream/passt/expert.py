import importlib
from typing import List

import torch
import torchaudio
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence


class UpstreamExpert(nn.Module):
    def __init__(self, name: str, refresh=False):
        super().__init__()
        self.resampler = torchaudio.transforms.Resample(16000, 32000)
        self.module = importlib.import_module(f".hear21passt.{name}", __package__)
        self.model = self.module.load_model()

    def get_downsample_rates(self, key=None):
        return int(50.0 / 1000.0 * 16000)

    def forward(self, wavs: List[torch.Tensor]):
        wavs = pad_sequence(wavs, batch_first=True)
        wavs = self.resampler(wavs)
        embs, timestamps = self.module.get_timestamp_embeddings(wavs, self.model)

        return {
            "hidden_states": [embs],
        }
