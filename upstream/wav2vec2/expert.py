# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/wav2vec2/expert.py ]
#   Synopsis     [ the wav2vec2 wrapper ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


import argparse
from typing import List
from packaging import version

import torch
import fairseq
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from omegaconf.dictconfig import DictConfig

from upstream.interfaces import UpstreamBase
from utility.helper import zero_mean_unit_var_norm


class UpstreamExpert(UpstreamBase):
    def __init__(self, ckpt, **kwargs):
        super().__init__(**kwargs)
        assert version.parse(fairseq.__version__) > version.parse(
            "0.10.2"
        ), "Please install the fairseq master branch."

        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt])
        self.model = model[0]
        self.wav_normalize = cfg.task.normalize

        # These options are only used for aligning representations between s3prl and huggingface
        # See utility/compare_wav2vec2.py
        self.apply_padding_mask = True
        self.numpy_wav_normalize = False

        if len(self.hooks) == 0:
            module_name = "self.model.encoder.layers"
            for module_id in range(len(eval(module_name))):
                self.add_hook(
                    f"{module_name}[{module_id}]",
                    lambda input, output: input[0].transpose(0, 1),
                )
            self.add_hook("self.model.encoder", lambda input, output: output[0])

    def forward(self, wavs):
        device = wavs[0].device
        if self.wav_normalize:
            if self.numpy_wav_normalize:
                wavs = zero_mean_unit_var_norm([wav.cpu().numpy() for wav in wavs])
                wavs = [torch.from_numpy(wav).to(device) for wav in wavs]
            else:
                wavs = [F.layer_norm(wav, wav.shape) for wav in wavs]

        wav_lengths = torch.LongTensor([len(wav) for wav in wavs]).to(device)
        wav_padding_mask = ~torch.lt(
            torch.arange(max(wav_lengths)).unsqueeze(0).to(device),
            wav_lengths.unsqueeze(1),
        )
        padded_wav = pad_sequence(wavs, batch_first=True)

        results = self.model.extract_features(
            padded_wav, wav_padding_mask if self.apply_padding_mask else None
        )
        return {"default": results["x"]}
