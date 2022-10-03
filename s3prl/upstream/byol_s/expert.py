from typing import Dict, List, Union

import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from ..interfaces import SAMPLE_RATE
from .serab_byols import serab


class UpstreamExpert(nn.Module):
    def __init__(
        self,
        ckpt: str = None,
        model_name: str = None,
        window_secs: float = 1,
        hop_secs: float = 0.05,
    ):
        super().__init__()
        self.model = serab.load_model(ckpt, model_name)
        self.frame_duration = window_secs * 1000
        self.hop_size = hop_secs * 1000

    def get_downsample_rates(self, key: str = None) -> int:
        return int(self.hop_size / 1000 * SAMPLE_RATE)

    def forward(self, wavs: List[Tensor]) -> Dict[str, Union[Tensor, List[Tensor]]]:
        padded_wavs = pad_sequence(wavs, batch_first=True)
        embeddings, timestamps = serab.get_timestamp_embeddings(
            padded_wavs, self.model, self.frame_duration, self.hop_size
        )
        return {
            "hidden_states": [embeddings],
        }
