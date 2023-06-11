import logging
from typing import Dict, List, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from .wav2vec2.model import wav2vec2_model

HIDDEN_DIM = 8
logger = logging.getLogger(__name__)

class UpstreamExpert(nn.Module):
    def __init__(self, ckpt: str = None, **kwargs):
        super().__init__()

        ckpt = torch.load(ckpt, map_location="cpu")
        self.model = wav2vec2_model(**ckpt["config"])
        result = self.model.load_state_dict(ckpt["state_dict"], strict=False)
        logger.info(f"missing: {result.missing_keys}, unexpected: {result.unexpected_keys}")
        logger.info(f"{sum(p.numel() for p in self.model.parameters())} params")


    def get_downsample_rates(self, key: str) -> int:
        """
        Since we do not do any downsampling in this example upstream
        All keys' corresponding representations have downsample rate of 1
        """
        return 320

    def forward(self, wavs: List[Tensor]) -> Dict[str, Union[Tensor, List[Tensor]]]:
        """
        When the returning Dict contains the List with more than one Tensor,
        those Tensors should be in the same shape to train a weighted-sum on them.
        """
        device = wavs[0].device
        lengths = torch.LongTensor([len(wav) for wav in wavs]).to(device)
        wavs = pad_sequence(wavs, batch_first=True)
        repres, _ = self.model.extract_features(wavs, lengths)

        return {
            "hidden_states": repres,
        }
