# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/npc/expert.py ]
#   Synopsis     [ the npc wrapper ]
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
#-------------#
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
#-------------#
from .npc import NPC
from .audio import create_transform


############
# CONSTANT #
############
EXAMPLE_FEAT_SEQLEN = 1000


###################
# UPSTREAM EXPERT #
###################
class UpstreamExpert(nn.Module):
    """
    The NPC wrapper
    """

    def __init__(self, ckpt, feature_selection, **kwargs):
        super(UpstreamExpert, self).__init__()
        self.feature_selection = feature_selection
        ckpt = torch.load(ckpt, map_location='cpu')
        config = ckpt['config']

        self.preprocessor, feat_dim = create_transform(config['data']['audio'])

        # init model structure
        self.model = NPC(feat_dim, **config['model']['paras'])
        
        # load pretrained-weights
        self.model.load_state_dict(ckpt['model'])
        self.output_dim = config['model']['paras']['hidden_size']

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
        features = [self.preprocessor(wav.unsqueeze(0)) for wav in wavs]
        feat_lengths = [len(feat) for feat in features]
        features = pad_sequence(features, batch_first=True)

        if 'unmasked' in self.feature_selection:
            n_layer_feat = int(self.feature_selection.split('-')[-1])
            features = self.model.get_unmasked_feat(features, n_layer_feat)
        elif self.feature_selection == 'masked':
            predicted_BxLxM, features = self.model(features, testing=not self.training)
        else:
            raise ValueError

        features = [f[:l] for f, l in zip(features, feat_lengths)]
        return features
