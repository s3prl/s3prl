import logging

import torch
import torchaudio
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from transformers import Speech2TextProcessor, SpeechT5Processor, Speech2TextForConditionalGeneration, SpeechT5ForSpeechToText

SAMPLE_RATE = 16000
EXAMPLE_SEC = 5

logger = logging.getLogger(__name__)


class UpstreamExpert(torch.nn.Module):
    def __init__(self, ckpt, **kwds):
        super().__init__()
        # self.model = Speech2TextForConditionalGeneration.from_pretrained(ckpt)
        # self.processor = Speech2TextProcessor.from_pretrained(ckpt)
        self.model = SpeechT5ForSpeechToText.from_pretrained(ckpt)
        self.processor = SpeechT5Processor.from_pretrained(ckpt)

    def get_downsample_rates(self, key: str = None) -> int:
        return  320

    def forward(self, wavs):
        device = wavs[0].device
        wavs = [wav.detach().cpu().numpy() for wav in wavs]
        input_features = self.processor(
            audio=wavs,
            return_tensors="pt",
            padding=True,
            sampling_rate=SAMPLE_RATE,
        ).to(device)
        output_values = self.model.base_model.encoder(**input_features, output_hidden_states=True)
        return {"hidden_states": output_values.hidden_states}