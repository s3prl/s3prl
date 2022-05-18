# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/cpc/expert.py ]
#   Synopsis     [ the cpc wrapper ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


import argparse

import torch
from torch.nn.utils.rnn import pad_sequence

from ..interfaces import UpstreamBase
from .cpc_default_config import get_default_cpc_config
from .feature_loader import getAR, getEncoder, loadArgs
from .model import CPCModel as cpcmodel

SAMPLE_RATE = 16000
EXAMPLE_SEC = 3
EXAMPLE_BATCH_SIZE = 32


class UpstreamExpert(UpstreamBase):
    def __init__(self, ckpt, **kwargs):
        super().__init__(**kwargs)

        locArgs = get_default_cpc_config()
        checkpoint = torch.load(ckpt, map_location="cpu")
        loadArgs(locArgs, argparse.Namespace(**checkpoint["config"]))

        encoderNet = getEncoder(locArgs)
        arNet = getAR(locArgs)
        self.model = cpcmodel(encoderNet, arNet)
        self.model.load_state_dict(checkpoint["weights"], strict=False)

        if len(self.hooks) == 0:
            self.add_hook(
                "self.model.gEncoder", lambda input, output: output.transpose(1, 2)
            )
            self.add_hook("self.model.gAR", lambda input, output: output)

    def get_downsample_rates(self, key: str) -> int:
        return 160

    def forward(self, wavs):
        padded_wav = pad_sequence(wavs, batch_first=True)
        features = self.model(padded_wav.unsqueeze(1), None)[0]

        # This forward function only does the model forward
        # The return dict is then handled by UpstreamBase's hooks
