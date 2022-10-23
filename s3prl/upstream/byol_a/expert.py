# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/byol_a/expert.py ]
#   Synopsis     [ the BYOL-Audio wrapper ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


import logging

import torch
import torch.nn as nn
import torchaudio

from .byol_a import AudioNTT2020, PrecomputedNorm, load_yaml_config

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


class UpstreamExpert(nn.Module):
    """
    The BYOL-A wrapper
    """

    def __init__(
        self,
        ckpt: str,
        model_config: str,
        feature_d: int,
        window_secs: float = 1.0,
        stride_secs: float = 1.0,
    ):
        super().__init__()
        config = load_yaml_config(model_config)

        self.window_secs = window_secs
        self.stride_secs = stride_secs
        self.output_dim = feature_d
        self.seg_input_length = len(
            range(0, int(window_secs * SAMPLE_RATE), config.hop_length)
        )

        # Preprocessor and normalizer.
        self.to_melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            f_min=config.f_min,
            f_max=config.f_max,
        )
        stats = [
            -5.4919195,
            5.0389895,
        ]  # FIXME: should use downstream dataset statistics
        self.normalizer = PrecomputedNorm(stats)

        # Load pretrained weights.
        self.model = AudioNTT2020(d=feature_d)
        self.model.load_weight(ckpt, device="cpu")

    def get_downsample_rates(self, key: str = None) -> int:
        return int(self.stride_secs * SAMPLE_RATE)

    def forward(self, wavs):
        """
        Args:
            wavs:
                list of unpadded wavs [wav1, wav2, ...]
                each wav is in torch.FloatTensor with sample rate 16000
                and already put in the device assigned by command-line args

        Return:
            features:
                list of unpadded features [feat1, feat2, ...]
                each feat is in torch.FloatTensor and already
                put in the device assigned by command-line args
        """
        wavs_len = [len(wav) for wav in wavs]
        max_wav_len = max(wavs_len)
        start_points = list(range(0, max_wav_len, int(self.stride_secs * SAMPLE_RATE)))
        padded_max_wav_len = start_points[-1] + int(self.window_secs * SAMPLE_RATE)
        padded_wavs = [
            torch.cat([wav, wav.new_zeros(padded_max_wav_len - len(wav))])
            for wav in wavs
        ]

        all_features = []
        for start in start_points:
            subwavs = [
                wav[start : start + int(self.window_secs * SAMPLE_RATE)]
                for wav in padded_wavs
            ]
            features = [
                self.normalizer(
                    (self.to_melspec(wav) + torch.finfo(torch.float).eps).log()
                ).permute(1, 0)
                for wav in subwavs
            ]
            features = torch.stack(
                features, dim=0
            )  # (batch_size, segment_seq_len, hiddqen_size)
            all_features.append(features)

        all_features = torch.stack(all_features, dim=0)
        num_segment, batch_size, segment_seq_len, hidden_size = all_features.shape

        flatten_features = all_features.reshape(-1, segment_seq_len, hidden_size)

        repre = self.model(
            flatten_features.transpose(1, 2).unsqueeze(1)
        )  # repre: (num_segment * batch_size, hidden_size)
        repre = repre.reshape(num_segment, batch_size, -1).transpose(
            0, 1
        )  # repre: (batch_size, num_segment, hidden_size)

        trimmed_hs = []
        for h in [repre]:
            max_h_len = len(range(0, max_wav_len, self.get_downsample_rates()))
            h = h[:, :max_h_len, :]
            trimmed_hs.append(h)

        return {
            "hidden_states": trimmed_hs,
        }
