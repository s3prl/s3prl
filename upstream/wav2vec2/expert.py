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


class UpstreamExpert(UpstreamBase):
    def __init__(self, ckpt, **kwargs):
        super().__init__(**kwargs)
        assert version.parse(fairseq.__version__) >= version.parse("0.10.2")

        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt])
        self.model = model[0]

        if isinstance(cfg, argparse.Namespace):
            normalize = cfg.normalize
        elif isinstance(cfg, DictConfig):
            normalize = cfg.task.normalize
        self.wav_normalize = normalize

        if len(self.hooks) == 0:
            module_name = "self.model.encoder.layers"
            for module_id in range(len(eval(module_name))):
                self.add_hook(
                    f"{module_name}[{module_id}]",
                    lambda input, output: input[0].transpose(0, 1),
                )
            self.add_hook("self.model.encoder", lambda input, output: output)

    @staticmethod
    def zero_mean_unit_var_norm(input_values: List[np.ndarray]) -> List[np.ndarray]:
        """
        Every array in the list is normalized to have zero mean and unit variance
        Taken from huggingface to ensure the same behavior across s3prl and huggingface
        Reference: https://github.com/huggingface/transformers/blob/a26f4d620874b32d898a5b712006a4c856d07de1/src/transformers/models/wav2vec2/feature_extraction_wav2vec2.py#L81-L86
        """
        return [(x - np.mean(x)) / np.sqrt(np.var(x) + 1e-5) for x in input_values]

    def forward(self, wavs):
        device = wavs[0].device
        if self.wav_normalize:
            wavs = self.zero_mean_unit_var_norm([wav.cpu().numpy() for wav in wavs])
            wavs = [torch.from_numpy(wav).to(device) for wav in wavs]

        wav_lengths = torch.LongTensor([len(wav) for wav in wavs]).to(device)
        wav_padding_mask = ~torch.lt(
            torch.arange(max(wav_lengths)).unsqueeze(0).to(device),
            wav_lengths.unsqueeze(1),
        )
        padded_wav = pad_sequence(wavs, batch_first=True)

        features, feat_padding_mask = self.model.extract_features(
            padded_wav, wav_padding_mask
        )
        return {"default": features}
