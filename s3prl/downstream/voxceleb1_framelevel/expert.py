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
import torch
import random
#-------------#
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
#-------------#
from argparse import Namespace
from ..voxceleb1.expert import  DownstreamExpert as SpeakerExpert

class DownstreamExpert(SpeakerExpert):
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log
    """

    def __init__(self, upstream_dim, downstream_expert, expdir, **kwargs):
        super(DownstreamExpert, self).__init__(upstream_dim, downstream_expert, expdir, **kwargs)
    # Interface
    def forward(self, mode, features, lengths, labels, records, **kwargs):
        device = features[0].device
        features_len = torch.IntTensor([len(feat) for feat in features]).to(device=device)
        features = pad_sequence(features, batch_first=True)
        features = self.projector(features)
        predicted, _ = self.model(features, features_len)

        labels = torch.LongTensor(labels).to(features.device)

        predicted = predicted.transpose(-1,-2)
        labels = [labels[index].expand(features_len[index]) for index in range(len(labels))]
        labels = pad_sequence(labels, padding_value=-100, batch_first=True)

        loss = self.objective(predicted, labels)

        predicted = predicted.transpose(-1,-2)
        predicted_classid = predicted.max(dim=-1).indices
        records['acc'] += (predicted_classid[labels!=-100] == labels[labels!=-100]).view(-1).cpu().float().tolist()
        records['loss'].append(loss.item())

        return loss
