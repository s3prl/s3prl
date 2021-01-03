import os
import sys
import math
import torch
import random

import yaml
import torch
import torch.nn as nn

from .audio import create_transform


class UpstreamExpert(nn.Module):
    """
    Pre-trained weights should be loaded
    """

    def __init__(self, config, *args, **kwargs):
        super(UpstreamExpert, self).__init__()
        
        with open(config, 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        audio = config['data']['audio']

        self.audio_transform, self.feat_dim = create_transform(
            audio.copy(),
            post_process=True,
            mode='train',
            read_audio=False,
        )

    # Interface
    def get_output_dim(self):
        return self.feat_dim

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
        features = [self.audio_transform(wav.view(1, -1)) for wav in wavs]
        return features
