# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/mockingjay/expert.py ]
#   Synopsis     [ the mockingjay wrapper ]
#   Author       [ Andy T. Liu (https://github.com/andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import yaml
#-------------#
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
#-------------#
from .builder import PretrainedTransformer


###################
# UPSTREAM EXPERT #
###################
class UpstreamExpert(nn.Module):
    """
    The Mockingjay wrapper
    """

    def __init__(self, ckpt, feature_selection=-1, model_config=None, **kwargs):
        super(UpstreamExpert, self).__init__()

        if model_config is not None:
            print('[UpstreamExpert] - Using upstream expert config file from:', model_config) 
            with open(model_config, 'r') as file:
                options = yaml.load(file, Loader=yaml.FullLoader)
        else:
            print('[UpstreamExpert] - Using the default upstream expert config') 
            options = {'load_pretrain' : 'True',
                       'no_grad'       : 'False',
                       'dropout'       : 'default',
                       'spec_aug'      : 'False',
                       'spec_aug_prev' : 'True',
                       'weighted_sum'  : 'False',
                       'permute_input' : 'False' }

        options['ckpt_file'] = ckpt
        options['select_layer'] = int(feature_selection)

        self.transformer = PretrainedTransformer(options, inp_dim=-1)
        assert hasattr(self.transformer, 'extracter'), 'This wrapper only supports `on-the-fly` ckpt with built in feature extracters.'

    # Interface
    def get_output_dim(self):
        return self.transformer.out_dim

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

        features = self.transformer(wavs) # (batch_size, extracted_seqlen, feature_dim)

        ratio = len(features[0]) / wav_lengths[0]
        feat_lengths = [round(l * ratio) for l in wav_lengths]
        features = [f[:l] for f, l in zip(features, feat_lengths)]

        return features