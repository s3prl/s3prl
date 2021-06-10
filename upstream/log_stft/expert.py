# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/log_stft/expert.py ]
#   Synopsis     [ the wrapper for STFT magnitude ]
#   Author       [ Zili Huang ]
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Spectrogram(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(Spectrogram, self).__init__()
        self.eps = 1e-8
        self.cfg = cfg
        self.n_fft = cfg['spectrogram']['n_fft']
        self.hop_length = cfg['spectrogram']['hop_length']
        self.win_length = cfg['spectrogram']['win_length']
        if cfg['spectrogram']['window'] == 'hann':
            self.window = torch.hann_window(cfg['spectrogram']['win_length']).to(device)
        else:
            raise ValueError("Window type not defined.")
        self.center = cfg['spectrogram']['center']
        self.log = cfg['spectrogram']['log']

    def get_output_dim(self):
        return self.n_fft // 2 + 1

    def get_downsample_rate(self):
        return self.hop_length

    def forward(self, waveform):
        # waveform: (time, )
        x = torch.transpose(torch.abs(torch.stft(waveform,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                center=self.center,
                pad_mode='reflect',
                normalized=False,
                return_complex=True)), 0, 1)
        if self.log:
            x = torch.log(torch.clamp(x, min=self.eps))
        # x: (feat_seqlen, feat_dim)
        return x

###################
# UPSTREAM EXPERT #
###################
class UpstreamExpert(nn.Module):
    """
    Extract spectrogram features from wavforms with torchaudio
    """

    def __init__(self, model_config=None, **kwargs):
        super(UpstreamExpert, self).__init__()

        with open(model_config, 'r') as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)
        self.extracter = Spectrogram(self.config)
        self.output_dim = self.extracter.get_output_dim()
        self.downsample_rate = self.extracter.get_downsample_rate()

    # Interface
    def get_output_dim(self):
        return self.output_dim

    # Interface
    def get_downsample_rate(self):
        return self.downsample_rate

    def _extractor_forward(self, wavs):
        feats = []
        for wav in wavs:
            feats.append(self.extracter(wav))
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
        feats = self._extractor_forward(wavs)
        return feats
