import logging

import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

SAMPLE_RATE = 16000
EXAMPLE_SEC = 5

logger = logging.getLogger(__name__)


class UpstreamExpert(torch.nn.Module):
    def __init__(self, ckpt, **kwds):
        super().__init__()
        self.extracter = Wav2Vec2FeatureExtractor.from_pretrained(ckpt)
        self.model = Wav2Vec2Model.from_pretrained(ckpt)

    def get_downsample_rates(self, key: str = None) -> int:
        return 320

    def forward(self, wavs):
        device = wavs[0].device
        wavs = [wav.detach().cpu().numpy() for wav in wavs]
        input_values = self.extracter(
            wavs,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
            sampling_rate=SAMPLE_RATE,
        ).to(device)
        output_values = self.model(**input_values, output_hidden_states=True)

        return {"hidden_states": output_values.hidden_states}
