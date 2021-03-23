# Copyright (c) Facebook, Inc. All Rights Reserved

# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/vq_wav2vec/expert.py ]
#   Synopsis     [ the vq_wav2vec wrapper ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import math
import yaml
import random
from packaging import version
#-------------#
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
#-------------#
import fairseq
from fairseq.models.wav2vec import Wav2VecModel


############
# CONSTANT #
############
SAMPLE_RATE = 16000
EXAMPLE_SEC = 5


###################
# UPSTREAM EXPERT #
###################
class UpstreamExpert(nn.Module):
    """
    The vq-wav2vec wrapper
    """

    def __init__(self, ckpt, feature_selection='z', **kwargs):
        super(UpstreamExpert, self).__init__()
        self.feature_selection = feature_selection or 'z'

        if version.parse(fairseq.__version__) > version.parse("0.10.2"):
            model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt])
            self.model = model[0]
            self.model.eval()
        elif version.parse(fairseq.__version__) == version.parse("0.10.2"):
            cp = torch.load(ckpt)
            self.model = Wav2VecModel.build_model(cp['args'], task=None)
            self.model.load_state_dict(cp['model'])
        else:
            raise NotImplementedError

        pseudo_input = torch.randn(SAMPLE_RATE * EXAMPLE_SEC)
        pseudo_output = self.forward([pseudo_input])
        self.output_dim = pseudo_output[0].size(-1)

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
        wav_lengths = [len(wav) for wav in wavs]
        padded_wav = pad_sequence(wavs, batch_first=True)
        z = self.model.feature_extractor(padded_wav)
        if self.feature_selection != 'z':
            codewords, codeids = self.model.vector_quantizer.forward_idx(z)
            codewords = codewords.transpose(1, 2).contiguous()
        z = z.transpose(1, 2).contiguous()
        # z, codewords: (batch_size, seqlen, feat_dim)
        # codeids: (batch_size, seqlen, 2)
        
        features = eval(self.feature_selection)
        ratio = padded_wav.size(1) / features.size(1)
        feat_lengths = [round(wav_len / ratio) for wav_len in wav_lengths]

        features = [feat[:length] for feat, length in zip(features, feat_lengths)]
        return features
