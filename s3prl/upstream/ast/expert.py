# -*- coding: utf-8 -*-
# @Time    : 8/25/21 5:25 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : expert.py

import logging

import torch

from s3prl.upstream.interfaces import SAMPLE_RATE

from ..interfaces import SAMPLE_RATE
from ..ssast.audio import FeatureExtractor

logger = logging.getLogger(__name__)
STRIDE = 160


class UpstreamExpert(torch.nn.Module):
    def __init__(self, ckpt: str, segment_secs: float = 10.24):
        super().__init__()
        self.segment_secs = segment_secs
        target_length = int(segment_secs * SAMPLE_RATE / STRIDE)

        try:
            import timm

            from .ast_models import ASTModel
        except:
            print(
                "SSAST requires 'timm==0.4.5' to work. Please run 'pip install timm==0.4.5'"
            )
            raise

        self.preprocessor = FeatureExtractor(
            target_length=target_length, apply_cmvn=False
        )
        self.model = ASTModel(
            label_dim=527,
            fstride=10,
            tstride=10,
            input_fdim=128,
            input_tdim=1024,
            imagenet_pretrain=True,
            audioset_pretrain=False,
            model_size="base384",
            pretrained_ckpt=ckpt,
        )

    def get_downsample_rates(self, key: str = None) -> int:
        return int(self.segment_secs * SAMPLE_RATE)

    def forward(self, wavs):
        wavs_len = [len(wav) for wav in wavs]
        max_wav_len = max(wavs_len)
        seg_n_sample = int(self.segment_secs * SAMPLE_RATE)
        padded_max_wav_len = (max_wav_len // seg_n_sample + 1) * seg_n_sample
        padded_wavs = [
            torch.cat([wav, wav.new_zeros(padded_max_wav_len - len(wav))])
            for wav in wavs
        ]

        all_ts = []
        for start in range(0, max(len(wav) for wav in padded_wavs), seg_n_sample):
            subwavs = [wav[start : start + seg_n_sample] for wav in padded_wavs]
            features = [self.preprocessor(wav.unsqueeze(0)) for wav in subwavs]
            features = torch.stack(features, dim=0)
            output = self.model(features)  # (batch_size, hidden_size)
            all_ts.append(output)

        all_hs = torch.stack(all_ts, dim=1)
        max_feat_len = int(max_wav_len // self.get_downsample_rates() + 1)
        all_hs = all_hs[:, :max_feat_len, :].float()

        return {"hidden_states": [all_hs]}
