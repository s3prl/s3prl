# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ expert.py ]
#   Synopsis     [ the pase wrapper ]
#   Author       [ santi-pdp/pase ]
#   Reference    [ https://github.com/santi-pdp/pase/blob/master/pase ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
#-------------#
from pase.models.frontend import wf_builder


SAMPLE_RATE = 16000
EXAMPLE_SEC = 5


####################
# UPSTREAM WRAPPER #
####################
class UpstreamExpert(nn.Module):
    """
    The PASE wrapper
    """

    def __init__(self, ckpt, config, **kwargs):
        super(UpstreamExpert, self).__init__() 

        self.pase = wf_builder(config)
        self.pase.load_pretrained(ckpt, load_last=True, verbose=False)

        # pseudo_input = torch.randn(1, 1, SAMPLE_RATE * EXAMPLE_SEC)
        # r = self.pase(pseudo_input) # size will be (1, 256, 625), which are 625 frames of 256 dims each
        self.output_dim = 256 # r.size(1)
        raise RuntimeError('There are some import errors with the PASE repo, see this issue: https://github.com/santi-pdp/pase/issues/114.')

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
        wav_lengths = [len(wav) for wav in wavs]
        wavs = pad_sequence(wavs, batch_first=True)
        wavs = wavs.unsqueeze(1) # (batch_size, audio_len) -> (batch_size, 1, audio_len)

        features = self.pase(wavs) # (batch_size, feature_dim, extracted_seqlen)
        features = features.permute(0, 2, 1).contiguous() # (batch_size, extracted_seqlen, feature_dim)

        ratio = len(features[0]) / wav_lengths[0]
        feat_lengths = [round(l * ratio) for l in wav_lengths]
        features = [f[:l] for f, l in zip(features, feat_lengths)]

        return features
