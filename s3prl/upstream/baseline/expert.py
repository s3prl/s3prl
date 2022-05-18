# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/baseline/expert.py ]
#   Synopsis     [ the baseline wrapper ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


import yaml
from torch.nn.utils.rnn import pad_sequence

from ..interfaces import UpstreamBase
from .extracter import get_extracter
from .preprocessor import get_preprocessor

SAMPLE_RATE = 16000

###################
# UPSTREAM EXPERT #
###################
class UpstreamExpert(UpstreamBase):
    """
    Extract baseline features from wavforms by torchaudio.compliance.kaldi or torchaudio preprocessor
    Support: spectrogram, fbank, mfcc, mel, linear
    """

    def __init__(self, model_config, **kwargs):
        super().__init__(**kwargs)

        with open(model_config, "r") as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)

        if "kaldi" in self.config:
            self.extracter, self.output_dim, frame_shift = get_extracter(self.config)
            self.downsample_rate = round(frame_shift * SAMPLE_RATE / 1000)
        else:
            self.extracter, self.output_dim, _ = get_preprocessor(
                self.config, process_input_only=True
            )
            self.downsample_rate = round(
                self.config.get("hop_ms", 10) * SAMPLE_RATE / 1000
            )

    def _extractor_forward(self, wavs):
        feats = []
        for wav in wavs:
            feats.append(self.extracter(wav))
        return feats

    def get_downsample_rates(self, key: str) -> int:
        return self.downsample_rate

    def _preprocessor_forward(self, wavs):
        wav_lengths = [len(wav) for wav in wavs]

        feats = pad_sequence(wavs, batch_first=True)
        feats = feats.unsqueeze(
            1
        )  # (batch_size, audio_len) -> (batch_size, 1, audio_len)
        feats = self.extracter(feats)[0]

        ratio = len(feats[0]) / wav_lengths[0]
        feat_lengths = [round(l * ratio) for l in wav_lengths]
        feats = [f[:l] for f, l in zip(feats, feat_lengths)]
        return feats

    def forward(self, wavs):
        if "kaldi" in self.config:
            feats = self._extractor_forward(wavs)
        else:
            feats = self._preprocessor_forward(wavs)

        padded_feats = pad_sequence(feats, batch_first=True)
        return {
            "last_hidden_state": padded_feats,
            "hidden_states": [padded_feats],
        }
