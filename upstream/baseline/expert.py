# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/baseline/expert.py ]
#   Synopsis     [ the baseline wrapper ]
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
from .extracter import get_extracter
from .preprocessor import get_preprocessor


###################
# UPSTREAM EXPERT #
###################
class UpstreamExpert(nn.Module):
    """
    Extract baseline features from wavforms by torchaudio.compliance.kaldi or torchaudio preprocessor
    Support: spectrogram, fbank, mfcc, mel, linear
    """

    def __init__(self, config, **kwargs):
        super(UpstreamExpert, self).__init__()

        with open(config, 'r') as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)

        if 'kaldi' in self.config:
            self.extracter, self.output_dim = get_extracter(self.config)
        else:
            self.extracter, self.output_dim, _ = get_preprocessor(self.config, process_input_only=True)

    # Interface
    def get_output_dim(self):
        return self.output_dim

    # Interface
    def get_downsample_rate(self):
        return 160

    def _extractor_forward(self, wavs):
        feats = []
        for wav in wavs:
            feats.append(self.extracter(wav))
        return feats

    def _preprocessor_forward(self, wavs):
        wav_lengths = [len(wav) for wav in wavs]
        
        feats = pad_sequence(wavs, batch_first=True)
        feats = feats.unsqueeze(1) # (batch_size, audio_len) -> (batch_size, 1, audio_len)
        feats = self.extracter(feats)[0]
        
        ratio = len(feats[0]) / wav_lengths[0]
        feat_lengths = [round(l * ratio) for l in wav_lengths]
        feats = [f[:l] for f, l in zip(feats, feat_lengths)]
        return feats

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
        if 'kaldi' in self.config:
            feats = self._extractor_forward(wavs)
        else:
            feats = self._preprocessor_forward(wavs)
        return feats
