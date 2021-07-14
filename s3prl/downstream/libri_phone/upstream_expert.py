import os
import math
import yaml
import torch
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from s3prl import hub
from .model import *

EXAMPLE_FEAT_SEQLEN = 1000
PHONE_CLASSES = 41
LABEL_STRIDE = 160


class UpstreamExpert(nn.Module):
    def __init__(self, ckpt, **kwargs):
        super(UpstreamExpert, self).__init__()
        ckpt = torch.load(ckpt, map_location='cpu')

        args = ckpt['Args']
        self.upstream = getattr(hub, args.upstream)()

        config = ckpt['Config']
        modelrc = config['downstream_expert']['modelrc']
        model_cls = eval(modelrc['select'])
        model_conf = modelrc[modelrc['select']]
        self.model = model_cls(self.upstream.get_output_dim(), output_class_num=PHONE_CLASSES, **model_conf)
        self.model.load_state_dict(UpstreamExpert._fix_state_key(ckpt['Downstream']))

    @staticmethod
    def _fix_state_key(states):
        keys = list(states.keys())
        for key in keys:
            new_key = '.'.join(key.split('.')[1:])
            states[new_key] = states[key]
            states.pop(key)
        return states

    # Interface
    def get_output_dim(self):
        return PHONE_CLASSES

    # Interface
    def get_downsample_rate(self):
        return LABEL_STRIDE

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
        feats = self.upstream(wavs)
        feats_length = [len(f) for f in feats]

        feats = pad_sequence(feats, batch_first=True)
        posteriors = self.model(feats)
        posteriors = [F.softmax(p[:l], dim=-1) for p, l in zip(posteriors, feats_length)]

        return posteriors
