import importlib
from typing import List

import torch
import torch.nn as nn
import torchaudio
from torch.nn.utils.rnn import pad_sequence

SAMPLE_RATE = 16000


class UpstreamExpert(nn.Module):
    def __init__(
        self,
        name: str,
        refresh=False,
        window_secs: float = 0.16,
        stride_secs: float = 0.05,
    ):
        super().__init__()
        self.resampler = torchaudio.transforms.Resample(16000, 32000)
        self.module = importlib.import_module(f".hear21passt.{name}", __package__)
        self.model = self.module.load_model(
            timestamp_window=window_secs * 1000,
            timestamp_hop=stride_secs * 1000,
        )
        self.stride_secs = stride_secs

    def get_downsample_rates(self, key=None):
        return int(self.stride_secs * SAMPLE_RATE)

    def forward(self, wavs: List[torch.Tensor]):
        wavs = pad_sequence(wavs, batch_first=True)
        wavs = self.resampler(wavs)
        embs, timestamps = self.module.get_timestamp_embeddings(wavs, self.model)

        return {
            "hidden_states": [embs],
        }
