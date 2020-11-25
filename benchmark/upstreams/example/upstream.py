import os
import math
import torch
import random

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

MEL_SPEC_DIM = 80
EXAMPLE_OUTPUT_DIM = 128
EXAMPLE_WAV_LEN = 1000


class Upstream(nn.Module):
    """
    Pre-trained weights should be loaded
    """

    def __init__(self, ckpt, config, **kwargs):
        super(Upstream, self).__init__()
        self.output_dim = EXAMPLE_OUTPUT_DIM
        self.linear = nn.Linear(MEL_SPEC_DIM, self.output_dim)

    # Interface
    def get_output_dim(self):
        return self.output_dim

    # Interface
    def forward(self, wavs):
        """
        Args:
            wavs: list of unpadded wavs

        Return:
            features: (batch_size, extracted_seqlen, feature_dim)        
        """

        wavs = pad_sequence(wavs, batch_first=True)
        mels = torch.randn(len(wavs), EXAMPLE_WAV_LEN, MEL_SPEC_DIM).to(wavs.device)
        features = self.linear(mels)
        return features
