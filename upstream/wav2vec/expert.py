# Copyright (c) Facebook, Inc. All Rights Reserved

# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/wav2vec/expert.py ]
#   Synopsis     [ the wav2vec wrapper ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


import argparse
from packaging import version

import torch
from torch.nn.utils.rnn import pad_sequence

import fairseq
from fairseq.models.wav2vec import Wav2VecModel
from omegaconf.dictconfig import DictConfig

from upstream.interfaces import UpstreamBase
from utility.helper import zero_mean_unit_var_norm

SAMPLE_RATE = 16000
EXAMPLE_SEC = 5


class UpstreamExpert(UpstreamBase):
    """
    The wav2vec wrapper
    """

    def __init__(self, ckpt, **kwargs):
        super().__init__(**kwargs)
        if version.parse(fairseq.__version__) > version.parse("0.10.2"):
            model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
                [ckpt]
            )
            self.model = model[0]
            self.model.eval()
        elif version.parse(fairseq.__version__) == version.parse("0.10.2"):
            cp = torch.load(ckpt)
            self.model = Wav2VecModel.build_model(cp["args"], task=None)
            self.model.load_state_dict(cp["model"])
        else:
            raise NotImplementedError

        if isinstance(cfg, argparse.Namespace):
            normalize = cfg.normalize
        elif isinstance(cfg, DictConfig):
            normalize = cfg.task.normalize
        assert isinstance(normalize, bool)
        self.wav_normalize = normalize

        if len(self.hooks) == 0:
            self.add_hook(
                "self.model.feature_extractor",
                lambda input, output: output.transpose(1, 2),
            )
            self.add_hook(
                "self.model.feature_aggregator",
                lambda input, output: output.transpose(1, 2),
            )
            module_name = "self.model.feature_aggregator.conv_layers"
            for conv_id in range(len(eval(module_name)) - 1):
                self.add_hook(
                    f"{module_name}[{conv_id + 1}]",
                    lambda input, output: input[0].transpose(1, 2),
                )

    def forward(self, wavs):
        """
        Code snippet modified from fairseq
        """
        device = wavs[0].device
        if self.wav_normalize:
            wavs = zero_mean_unit_var_norm([wav.cpu().numpy() for wav in wavs])
            wavs = [torch.from_numpy(wav).to(device) for wav in wavs]

        result = {}

        padded_wav = pad_sequence(wavs, batch_first=True)
        features = self.model.feature_extractor(padded_wav)
        result["z"] = features.transpose(1, 2).contiguous()

        if self.model.vector_quantizer:
            q_res = self.model.vector_quantizer(features, produce_targets=True)
            result["codewords"] = q_res["x"].transpose(1, 2).contiguous()
            result["codeids"] = q_res["targets"]
            features = q_res["x"]

        x = self.model.dropout_feats(features)
        x = self.model.feature_aggregator(x)

        result["c"] = x.transpose(1, 2).contiguous()
        result["default"] = result["c"]

        return result
