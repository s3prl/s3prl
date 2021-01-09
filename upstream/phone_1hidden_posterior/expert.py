# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ expert.py ]
#   Synopsis     [ the phone linear downstream wrapper ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import math
import random
#-------------#
import torch
import torch.nn as nn
#-------------#
from benchmark.downstream.phone_linear.model import Model

REPO = 'andi611/Self-Supervised-Speech-Pretraining-and-Representation-Learning:benchmark'
PHONE_CLASSES = 41


class UpstreamExpert(nn.Module):
    """
    Used to extract phoneme-posteriorgram as feature
    """

    def __init__(self, ckpt, **kwargs):
        super(UpstreamExpert, self).__init__()
        ckpt = torch.load(ckpt, map_location='cpu')

        self.upstream = torch.hub.load(REPO, ckpt['Args'].upstream,
            force_reload=True,
        )
        self.modelrc = ckpt['Config']['downstream_expert']['modelrc']
        self.model = Model(input_dim=self.upstream.get_output_dim(), output_class_num=PHONE_CLASSES, **self.modelrc)

    def get_output_dim(self):
        return PHONE_CLASSES

    # Interface
    def forward(self, wavs, **kwargs):
        """
        Args:
            features:
                list of unpadded features [feat1, feat2, ...]
                each feat is in torch.FloatTensor and already
                put in the device assigned by command-line args

        Return:
            features:
                list of unpadded features [feat1, feat2, ...]
                each feat is in torch.FloatTensor and already
                put in the device assigned by command-line args
        """
        features = self.upstream(wavs)
        lengths = torch.LongTensor([len(f) for f in features])
        length_masks = torch.lt(torch.arange(lengths.max()).unsqueeze(0), lengths.unsqueeze(-1))
        length_masks = length_masks.to(features[0].device)

        features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
        predicted = self.model(features) * length_masks.unsqueeze(-1)

        normalized = predicted / (predicted.sum(dim=-1, keepdim=True) + 1e-8)
        normalized = [n[:l] for n, l in zip(normalized, lengths)]

        return normalized
