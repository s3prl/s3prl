import os
import math
import torch
import random

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torchaudio.transforms import MelSpectrogram

SAMPLE_RATE = 16000
MEL_SPEC_DIM = 80
EXAMPLE_OUTPUT_DIM = 128


class UpstreamExpert(nn.Module):
    """
    Pre-trained weights should be loaded
    """

    def __init__(self, ckpt, config, **kwargs):
        super(UpstreamExpert, self).__init__()
        self.output_dim = EXAMPLE_OUTPUT_DIM
        self.melspectrogram = MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=MEL_SPEC_DIM)
        self.linear = nn.Linear(MEL_SPEC_DIM, self.output_dim)

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
        wav_lengths = [len(wav) for wav in wavs]
        wavs = pad_sequence(wavs, batch_first=True)

        mels = self.melspectrogram(wavs).transpose(1, 2)
        features = self.linear(mels)

        ratio = len(features[0]) / wav_lengths[0]
        feat_lengths = [round(l * ratio) for l in wav_lengths]
        features = [f[:l] for f, l in zip(features, feat_lengths)]

        return features
