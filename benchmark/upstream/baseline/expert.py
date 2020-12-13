import os
import math
import yaml
import torch
import random

import torch
import torch.nn as nn

from .extracter import get_extracter


class UpstreamExpert(nn.Module):
    """
    Extract baseline features from wavforms by torchaudio.compliance.kaldi
    Support: spectrogram, fbank, mfcc
    """

    def __init__(self, config, **kwargs):
        super(UpstreamExpert, self).__init__()

        with open(config, 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

        self.extracter, self.output_dim = get_extracter(config)

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
        feats = []
        for wav in wavs:
            feats.append(self.extracter(wav))

        return feats
