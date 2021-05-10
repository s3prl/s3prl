# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/pase/expert.py ]
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


############
# CONSTANT #
############
SAMPLE_RATE = 16000
EXAMPLE_SEC = 5


####################
# UPSTREAM WRAPPER #
####################
class UpstreamExpert(nn.Module):
    """
    The PASE wrapper
    """

    def __init__(self, ckpt, model_config, **kwargs):
        super(UpstreamExpert, self).__init__() 

        def build_pase(ckpt, model_config):
            pase = wf_builder(model_config)
            pase.load_pretrained(ckpt, load_last=True, verbose=False)
            return pase

        # pase can be only used on GPU as the official implementation
        pase = build_pase(ckpt, model_config).cuda()
        pseudo_input = torch.randn(1, 1, SAMPLE_RATE * EXAMPLE_SEC).cuda()
        self.output_dim = pase(pseudo_input).size(1)

        self.model = build_pase(ckpt, model_config)

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
                each wav is in torch.FloatTensor and already
                put in the device assigned by command-line args

        Return:
            features:
                (batch_size, extracted_seqlen, feature_dim)        
        """
        wav_lengths = [len(wav) for wav in wavs]
        wavs = pad_sequence(wavs, batch_first=True)
        wavs = wavs.unsqueeze(1)

        features = self.model(wavs) # (batch_size, feature_dim, extracted_seqlen)
        features = features.transpose(1, 2).contiguous() # (batch_size, extracted_seqlen, feature_dim)

        ratio = len(features[0]) / wav_lengths[0]
        feat_lengths = [round(l * ratio) for l in wav_lengths]
        features = [f[:l] for f, l in zip(features, feat_lengths)]

        return features
