import logging

import torch
from transformers import HubertModel, Wav2Vec2FeatureExtractor

SAMPLE_RATE = 16000
EXAMPLE_SEC = 5

logger = logging.getLogger(__name__)


class UpstreamExpert(torch.nn.Module):
    def __init__(self, ckpt, **kwds):
        super().__init__()
        try:
            self.extracter = Wav2Vec2FeatureExtractor.from_pretrained(ckpt)
        except:
            if "base" in ckpt:
                alter_extractor = "facebook/hubert-base-ls960"
            else:
                alter_extractor = "facebook/hubert-large-ll60k"
            logger.info(
                f"The model {ckpt} on huggingface does not have a correspoinding feature extractor. "
                f"Using {alter_extractor}'s feature extractor as the alternative."
            )
            self.extracter = Wav2Vec2FeatureExtractor.from_pretrained(alter_extractor)
        self.model = HubertModel.from_pretrained(ckpt)

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
