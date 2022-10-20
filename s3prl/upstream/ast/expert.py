# -*- coding: utf-8 -*-
# @Time    : 8/25/21 5:25 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : expert.py

import logging

import torch

from ..ssast.audio import FeatureExtractor

logger = logging.getLogger(__name__)

STRIDE = 160
SAMPLE_RATE = 16000


class UpstreamExpert(torch.nn.Module):
    def __init__(
        self, ckpt: str, window_secs: float = 10.24, stride_secs: float = 10.24
    ):
        super().__init__()
        self.window_secs = window_secs
        self.stride_secs = stride_secs
        target_length = int(window_secs * SAMPLE_RATE / STRIDE)

        try:
            import timm

            from .ast_models import ASTModel
        except:
            logger.error(
                "SSAST requires 'timm==0.4.5' to work. Please run 'pip install timm==0.4.5'"
            )
            exit(1)

        self.preprocessor = FeatureExtractor(
            target_length=target_length, apply_cmvn=False
        )
        self.model = ASTModel(
            label_dim=527,
            fstride=10,
            tstride=10,
            input_fdim=128,
            input_tdim=int(window_secs * 100),
            imagenet_pretrain=True,
            audioset_pretrain=True,
            model_size="base384",
            pretrained_ckpt=ckpt,
        ).cpu()  # ensure the entire model is on cpu

    def get_downsample_rates(self, key: str = None) -> int:
        return int(self.stride_secs * SAMPLE_RATE)

    def forward(self, wavs):
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
            features = [self.preprocessor(wav.unsqueeze(0)) for wav in subwavs]
            features = torch.stack(
                features, dim=0
            )  # (batch_size, segment_seq_len, hidden_size)
            all_features.append(features)

        all_features = torch.stack(all_features, dim=0)
        num_segment, batch_size, segment_seq_len, hidden_size = all_features.shape

        flatten_features = all_features.reshape(-1, segment_seq_len, hidden_size)
        output = self.model(flatten_features)  # (num_segment * batch_size, hidden_size)

        output = output.reshape(num_segment, batch_size, -1).transpose(0, 1).float()
        # (batch_size, num_segment, output_sizeq)

        return {
            "hidden_states": [output],
        }
