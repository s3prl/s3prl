from typing import List

import torch
import torch.nn as nn
from torch import Tensor
from transformers import Wav2Vec2Processor, Wav2Vec2Model

SAMPLE_RATE = 16000


class UpstreamExpert(nn.Module):
    def __init__(self, ckpt: str = None, model_config: str = None, **kwargs):
        super().__init__()

        self.processor = Wav2Vec2Processor.from_pretrained(ckpt)
        self.model = Wav2Vec2Model.from_pretrained(ckpt)

    def get_downsample_rates(self, key: str) -> int:
        return 320

    def forward(self, wavs: List[Tensor]):
        device = wavs[0].device
        processor_outputs = self.processor(
            [wav.cpu().numpy() for wav in wavs],
            return_tensors="pt",
            sampling_rate=SAMPLE_RATE,
            padding="longest",
        )
        attention_mask = processor_outputs.get("attention_mask", None)
        if isinstance(attention_mask, torch.Tensor):
            attention_mask = attention_mask.to(device)
        model_outputs = self.model(
            processor_outputs.input_values.to(device),
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        return {
            "last_hidden_state": model_outputs.last_hidden_state,
            "hidden_states": model_outputs.hidden_states,
        }
