# -*- coding: utf-8 -*-
# @Time    : 8/25/21 5:25 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : expert.py

# Author
# - Leo

import logging

import torch

from ..ssast.audio import FeatureExtractor

logger = logging.getLogger(__name__)

FBANK_SAMPLE_STRIDE = 160
PATCH_FBANK_STRIDE = 10
SAMPLE_RATE = 16000


class UpstreamExpert(torch.nn.Module):
    def __init__(
        self,
        ckpt: str,
        window_secs: float = 10.24,
        stride_secs: float = 10.24,
        feature_selection: str = "cls",
    ):
        super().__init__()
        assert feature_selection in ["cls", "hidden_states"]
        self.feature_selection = feature_selection

        self.window_secs = window_secs
        self.stride_secs = stride_secs
        target_length = int(window_secs * SAMPLE_RATE / FBANK_SAMPLE_STRIDE)

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
            tstride=PATCH_FBANK_STRIDE,
            input_fdim=128,
            input_tdim=int(window_secs * 100),
            imagenet_pretrain=True,
            audioset_pretrain=True,
            model_size="base384",
            pretrained_ckpt=ckpt,
        ).cpu()  # ensure the entire model is on cpu

    def get_downsample_rates(self, key: str = None) -> int:
        if self.feature_selection == "cls":
            return int(self.stride_secs * SAMPLE_RATE)
        elif self.feature_selection == "hidden_states":
            return int(FBANK_SAMPLE_STRIDE * PATCH_FBANK_STRIDE)

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
        output, hidden_states = self.model(flatten_features)
        # output: (num_segment * batch_size, num_class)
        # hidden_states: List[(num_segment * batch_size, segment_seq_len, hidden_size)]

        if self.feature_selection == "cls":
            output = output.reshape(num_segment, batch_size, -1).transpose(0, 1).float()
            # (batch_size, num_segment, num_class)
            hidden_states = [output]

        elif self.feature_selection == "hidden_states":
            reshaped_hidden_states = [
                (
                    h.reshape(num_segment, batch_size, -1, h.size(-1))
                    .transpose(
                        0, 1
                    )  # (batch_size, num_segment, num_horizon_patch, num_vertical_patch * hidden_size)
                    .flatten(
                        1, 2
                    )  # (batch_size, num_segment * num_horizon_patch, num_vertical_patch * hidden_size)
                    .float()
                )
                for h in hidden_states
            ]
            hidden_states = reshaped_hidden_states

        trimmed_hidden_states = []
        for h in hidden_states:
            max_h_len = len(range(0, max_wav_len, self.get_downsample_rates()))
            h = h[:, :max_h_len, :]
            trimmed_hidden_states.append(h)

        return {
            "hidden_states": trimmed_hidden_states,
        }
