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
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
#-------------#
from .builder import PretrainedTransformer


####################
# UPSTREAM WRAPPER #
####################
class UpstreamExpert(nn.Module):
    """
    The Mockingjay wrapper
    """

    def __init__(self, ckpt, config=None, **kwargs):
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

        self.transformer = PretrainedTransformer(options, inp_dim=-1)
        self.output_dim = self.transformer.out_dim
        assert hasattr(self.transformer, 'extracter'), 'This wrapper only supports `on-the-fly` ckpt with built in feature extracters.'

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

        features = self.transformer(wavs) # (batch_size, extracted_seqlen, feature_dim)

        ratio = len(features[0]) / wav_lengths[0]
        feat_lengths = [round(l * ratio) for l in wav_lengths]
        features = [f[:l] for f, l in zip(features, feat_lengths)]

        return features