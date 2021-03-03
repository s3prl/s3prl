# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/cpc/expert.py ]
#   Synopsis     [ the cpc wrapper ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import math
import yaml
import torch
import random
import argparse
#-------------#
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
#-------------#
from .model import CPCModel as cpcmodel
from .cpc_default_config import get_default_cpc_config
from .feature_loader import getEncoder, getAR, loadArgs


############
# CONSTANT #
############
SAMPLE_RATE = 16000
EXAMPLE_SEC = 3
EXAMPLE_BATCH_SIZE = 32


###################
# UPSTREAM EXPERT #
###################
class UpstreamExpert(nn.Module):
    """
    The CPC wrapper
    """

    def __init__(self, ckpt, **kwargs):
        super(UpstreamExpert, self).__init__()

        locArgs = get_default_cpc_config()
        checkpoint = torch.load(ckpt, map_location='cpu')
        loadArgs(locArgs, argparse.Namespace(**checkpoint["config"]))

        encoderNet = getEncoder(locArgs)
        arNet = getAR(locArgs)
        self.model = cpcmodel(encoderNet, arNet)
        self.model.load_state_dict(checkpoint["weights"], strict=False)

        pseudo_input = torch.randn(EXAMPLE_BATCH_SIZE, SAMPLE_RATE * EXAMPLE_SEC)
        pseudo_output = self.model(pseudo_input.unsqueeze(1), None)[0]
        self.output_dim = pseudo_output.size(-1)


    # Interface
    def get_output_dim(self):
        return self.output_dim

    # Interface
    def get_downsample_rate(self):
        return 160

    # Interface
    def forward(self, wavs):
        """
        Args:
            wavs:
                list of unpadded wavs [wav1, wav2, ...]
                each wav is in torch.FloatTensor with sample rate 16000
                and already put in the device assigned by command-line args

        Return:
            features:
                list of unpadded features [feat1, feat2, ...]
                each feat is in torch.FloatTensor and already
                put in the device assigned by command-line args
        """
        device = wavs[0].device
        wav_lengths = [len(wav) for wav in wavs]
        padded_wav = pad_sequence(wavs, batch_first=True)

        features = self.model(padded_wav.unsqueeze(1), None)[0]

        ratio = len(features[0]) / wav_lengths[0]
        feat_lengths = [round(l * ratio) for l in wav_lengths]
        features = [f[:l] for f, l in zip(features, feat_lengths)]

        return features
