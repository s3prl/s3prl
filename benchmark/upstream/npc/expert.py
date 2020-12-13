import os
import math
import yaml
import torch
import random

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from .npc import NPC
from .audio import create_transform

EXAMPLE_FEAT_SEQLEN = 1000


class UpstreamExpert(nn.Module):
    """
    The expert of NPC
    """

    def __init__(self, ckpt, config, **kwargs):
        super(UpstreamExpert, self).__init__()

        with open(config, 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

        self.preprocessor, feat_dim = create_transform(config['data']['audio'])

        # init model structure
        self.model = NPC(feat_dim, **config['model']['paras'])
        
        # load pretrained-weights
        ckpt = torch.load(ckpt, map_location='cpu')
        self.model.load_state_dict(ckpt['model'])

        pseudo_input = torch.randn(1, EXAMPLE_FEAT_SEQLEN, feat_dim)
        predicted_BxLxM, feature = self.model(pseudo_input, testing=True)
        self.output_dim = feature.size(-1)

    # Interface
    def get_output_dim(self):
        return self.output_dim

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
        features = [self.preprocessor(wav.unsqueeze(0)) for wav in wavs]
        feat_lengths = [len(feat) for feat in features]

        features = pad_sequence(features, batch_first=True)
        predicted_BxLxM, features = self.model(features, self.training)

        return features
