# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ expert.py ]
#   Synopsis     [ the mockingjay wrapper ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import math
import torch
import random
#-------------#
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
#-------------#
from transformer.nn_transformer import TRANSFORMER


####################
# UPSTREAM WRAPPER #
####################
class UpstreamExpert(nn.Module):
    """
    The Mockingjay wrapper
    """

    def __init__(self, ckpt, config, **kwargs):
        super(UpstreamExpert, self).__init__() 
        options = {'ckpt_file'     : ckpt,
                   'load_pretrain' : 'True',
                   'no_grad'       : 'True',
                   'dropout'       : 'default',
                   'spec_aug'      : 'False',
                   'spec_aug_prev' : 'True',
                   'weighted_sum'  : 'False',
                   'select_layer'  : -1,
                   'permute_input' : 'False' }

        self.transformer = TRANSFORMER(options, inp_dim=-1)
        self.output_dim = self.transformer.out_dim
        assert hasattr(self.transformer, 'preprocessor'), 'This wrapper only supports `on-the-fly` ckpt with built in preprocessors.'

    # Interface
    def get_output_dim(self):
        return self.output_dim

    # Interface
    def forward(self, wavs):
        """
        Args:
            wavs:
                list of unpadded wavs [wav1, wav2, ...]
                each wav is in torch.FloatTensor and already
                put in the device assigned by command-line args

        Return:
            features:
                (batch_size, extracted_seqlen, feature_dim)        
        """

        wavs = pad_sequence(wavs, batch_first=True)
        for i in range(wavs.size(0)): 
            wavs[i] = TRANSFORMER.normalize_wav_decibel(wavs[i], self.transformer.config['online']['target_level'])
        wavs = wavs.unsqueeze(-1) # (batch_size, audio_len) -> (batch_size, audio_len, 1)
        features = self.transformer(wavs) # (batch_size, extracted_seqlen, feature_dim)
        return features