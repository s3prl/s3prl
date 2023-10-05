# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
# 	Alan Baade originally adopted from Fast VGS Wrapper by Puyuan Peng,
#   later modified by the S3PRL team.
"""*********************************************************************************************"""
#   FileName     [ upstream/mae_ast/expert.py ]
#   Synopsis     [ Upstream MAE-AST Wrapper ]
#   Author       [ Alan Baade, S3PRL ]
#   Copyright    [ Copyleft(c), Alan Baade ]
"""*********************************************************************************************"""

from types import SimpleNamespace

import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence

from ..interfaces import UpstreamBase
from .mae_ast import MAE_AST

FBANK_SAMPLE_STRIDE = 160


class UpstreamExpert(UpstreamBase):
    def __init__(self, ckpt, **kwargs):
        super().__init__(**kwargs)

        checkpoint = torch.load(ckpt, map_location="cpu")

        self.cfg = checkpoint["cfg"]["model"]
        self.task_cfg = checkpoint["cfg"]["task"]

        self.model = MAE_AST(
            SimpleNamespace(**checkpoint["cfg"]["model"]),
            SimpleNamespace(**checkpoint["cfg"]["task"]),
        ).cpu()

        self.model.load_state_dict(checkpoint["model"], strict=True)

        # Required for hidden states to have defined indices.
        self.model.encoder.layerdrop = 0

        self.sample_rate = self.task_cfg["sample_rate"]
        self.feature_dim = self.task_cfg["feature_dim"]
        self.feature_rate = self.task_cfg["feature_rate"]

        self.ast_kernel_size_chan = self.cfg["ast_kernel_size_chan"]
        self.ast_kernel_size_time = self.cfg["ast_kernel_size_time"]
        self.ast_kernel_stride_chan = self.cfg["ast_kernel_stride_chan"]
        self.ast_kernel_stride_time = self.cfg["ast_kernel_stride_time"]
        self.num_vertical_patch = (
            self.feature_dim - self.ast_kernel_size_chan
        ) // self.ast_kernel_stride_chan + 1

        self.is_decoder_finetune = False  # TODO bad way of passing in info

    def get_downsample_rates(self, key: str = None) -> int:
        return int(self.ast_kernel_stride_time * FBANK_SAMPLE_STRIDE)

    def wav_to_spectrogram(self, wav):
        return torchaudio.compliance.kaldi.fbank(  # Frame shift and length are standard at 10, 25
            waveform=wav,
            sample_frequency=self.sample_rate,
            use_energy=False,
            num_mel_bins=self.feature_dim,
        )

    def forward(self, wavs):
        device = wavs[0].device

        wavs_len = [len(wav) for wav in wavs]
        features = [self.wav_to_spectrogram(wav.unsqueeze(0)) for wav in wavs]
        feature_lengths = torch.LongTensor([len(feature) for feature in features]).to(
            device
        )
        feature_padding_mask = ~torch.lt(
            torch.arange(max(feature_lengths)).unsqueeze(0).to(device),
            feature_lengths.unsqueeze(1),
        )
        padded_features = pad_sequence(features, batch_first=True)

        results = self.model(
            padded_features,
            padding_mask=feature_padding_mask,
            mask=False,
            features_only=True,
            is_decoder_finetune=self.is_decoder_finetune,
        )
        hidden_states = results["hidden_states"]

        finalized_hs = []
        for h in hidden_states:
            # h: (batch_size, num_horizon_patch * num_vertical_patch, hidden_size)
            h = h.reshape(h.size(0), -1, self.num_vertical_patch * h.size(-1))
            h = h[:, : len(range(0, max(wavs_len), self.get_downsample_rates())), :]
            finalized_hs.append(h)

        return {
            "hidden_states": finalized_hs,
        }
