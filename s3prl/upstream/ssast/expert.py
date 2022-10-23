# -*- coding: utf-8 -*-
# @Time    : 8/25/21 5:25 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : expert.py

# Authors
# - Leo

import logging

import torch

from .audio import FeatureExtractor

logger = logging.getLogger(__name__)

FBANK_SAMPLE_STRIDE = 160
SAMPLE_RATE = 16000


class UpstreamExpert(torch.nn.Module):
    def __init__(
        self,
        ckpt: str,
        model_size: str,
        window_secs: float = 1.0,
    ):
        super().__init__()
        self.window_secs = window_secs
        self.stride_secs = window_secs

        try:
            import timm

            from .ast_models import ASTModel
        except:
            logger.error(
                "SSAST requires 'timm==0.4.5' to work. Please run 'pip install timm==0.4.5'"
            )
            exit(1)

        target_length = int(window_secs * SAMPLE_RATE / FBANK_SAMPLE_STRIDE)
        model_size, model_type = model_size.split("_")[0], model_size.split("_")[1]
        self.preprocessor = FeatureExtractor(
            target_length=target_length, apply_cmvn=False
        )

        assert model_type in ["p", "f"]
        if model_type == "p":
            self.tstride = 10
            self.model = ASTModel(
                fshape=16,
                tshape=16,
                fstride=10,
                tstride=self.tstride,
                input_tdim=target_length,
                input_fdim=128,
                model_size=model_size,
                pretrain_stage=False,
                load_pretrained_mdl_path=ckpt,
            )
            self.vertical_num_patches = (128 - 16) // 10 + 1  # 12
        else:
            self.tstride = 1
            self.model = ASTModel(
                fshape=128,
                tshape=2,
                fstride=128,
                tstride=self.tstride,
                input_tdim=target_length,
                input_fdim=128,
                model_size=model_size,
                pretrain_stage=False,
                load_pretrained_mdl_path=ckpt,
            )
            self.vertical_num_patches = (128 - 128) // 128 + 1  # 1

        self.model = self.model.cpu()

    def get_downsample_rates(self, key: str = None) -> int:
        return int(FBANK_SAMPLE_STRIDE * self.tstride)

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
        hidden_states, final_repr = self.model(flatten_features)

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

        trimmed_hidden_states = []
        for h in reshaped_hidden_states:
            max_h_len = len(range(0, max_wav_len, self.get_downsample_rates()))
            h = h[:, :max_h_len, :]
            trimmed_hidden_states.append(h)

        return {"hidden_states": trimmed_hidden_states}
