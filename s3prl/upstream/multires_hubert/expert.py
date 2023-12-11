# Copyright (c) Facebook, Inc. All Rights Reserved

# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/multires_hubert/expert.py ]
#   Synopsis     [ the Multiresolution HuBERT wrapper ]
#   Author       [ Jiatong Shi ]
"""*********************************************************************************************"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from ..interfaces import UpstreamBase
from .convert import load_converted_model

SAMPLE_RATE = 16000
EXAMPLE_SEC = 5

logger = logging.getLogger(__name__)


def upsample(x, upsample_rate):
    return x.repeat_interleave(upsample_rate, dim=1)


class UpstreamExpert(UpstreamBase):
    def __init__(self, ckpt, **kwargs):
        super().__init__(**kwargs)
        model, task_cfg = load_converted_model(ckpt)
        self.model = model
        self.task_cfg = task_cfg
        self.final_factors = []

        self.num_res = self.model.label_nums
        logger.info("num res: {}".format(self.num_res))

        # decide ratios
        feature_ds_rates = self.model.feature_ds_rates
        lcm = np.lcm.reduce(feature_ds_rates)
        self.upsample_factor = [lcm // res for res in feature_ds_rates][::-1]
        self.reverse_upsample_factor = self.upsample_factor[::-1][1:]
        logger.info("upsample_factor: {}".format(self.upsample_factor))

        if len(self.hooks) == 0:
            # Process Encoders
            for res_index in range(self.num_res - 1):
                module_name = "self.model.encoders[{}].layers".format(res_index)
                for module_id in range(len(eval(module_name))):
                    self.add_hook(
                        f"{module_name}[{module_id}]",
                        lambda input, output: input[0].transpose(0, 1),
                    )
                    self.final_factors.append(self.upsample_factor[res_index])
                self.add_hook(
                    "self.model.encoders[{}]".format(res_index),
                    lambda input, output: output[0],
                )
                self.final_factors.append(self.upsample_factor[res_index])

            # Process middle encoders
            module_name = "self.model.middle_encoder.layers"
            for module_id in range(len(eval(module_name))):
                self.add_hook(
                    f"{module_name}[{module_id}]",
                    lambda input, output: input[0].transpose(0, 1),
                )
                self.final_factors.append(self.upsample_factor[self.num_res - 1])
            self.add_hook("self.model.middle_encoder", lambda input, output: output[0])
            self.final_factors.append(self.upsample_factor[self.num_res - 1])

            # Process decoders
            for res_index in range(self.num_res - 1):
                module_name = "self.model.decoders[{}].layers".format(res_index)
                for module_id in range(len(eval(module_name))):
                    self.add_hook(
                        f"{module_name}[{module_id}]",
                        lambda input, output: input[0].transpose(0, 1),
                    )
                    self.final_factors.append(self.reverse_upsample_factor[res_index])
                self.add_hook(
                    "self.model.decoders[{}]".format(res_index),
                    lambda input, output: output[0],
                )
                self.final_factors.append(self.reverse_upsample_factor[res_index])

            def postprocess(xs):
                names, hiddens = zip(*xs)
                assert len(hiddens) == len(self.final_factors)
                hiddens = [
                    upsample(hiddens[i], self.final_factors[i])
                    for i in range(len(hiddens))
                ]
                unpad_len = min([hidden.size(1) for hidden in hiddens])
                hiddens = [hidden[:, :unpad_len, :] for hidden in hiddens]
                return list(zip(names, hiddens))

            self.hook_postprocess = postprocess

    def get_downsample_rates(self, key: str) -> int:
        return 320

    def forward(self, wavs):
        if self.task_cfg.normalize:
            wavs = [F.layer_norm(wav, wav.shape) for wav in wavs]

        device = wavs[0].device
        wav_lengths = torch.LongTensor([len(wav) for wav in wavs]).to(device)
        wav_padding_mask = ~torch.lt(
            torch.arange(max(wav_lengths)).unsqueeze(0).to(device),
            wav_lengths.unsqueeze(1),
        )
        padded_wav = pad_sequence(wavs, batch_first=True)

        features, feat_padding_mask = self.model.extract_features(
            padded_wav,
            padding_mask=wav_padding_mask,
            mask=None,
        )

        # This forward function only does the model forward
        # The return dict is then handled by UpstreamBase's hooks
