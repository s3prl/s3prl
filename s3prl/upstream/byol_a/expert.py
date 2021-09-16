# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/byol_a/expert.py ]
#   Synopsis     [ the BYOL-Audio wrapper ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import math
#-------------#
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
#-------------#
import torchaudio
#-------------#
from .byol_a import load_yaml_config, PrecomputedNorm, AudioNTT2020


###################
# UPSTREAM EXPERT #
###################
class UpstreamExpert(nn.Module):
    """
    The BYOL-A wrapper
    """

    def __init__(self, ckpt, model_config=None, **kwargs):
        super(UpstreamExpert, self).__init__()

        if model_config is not None:
            print('[UpstreamExpert] - Using upstream expert config file from:', model_config) 
        else:
            model_config = './upstream/byol_a/config.yaml'
        config = load_yaml_config(model_config)

        # Preprocessor and normalizer.
        self.to_melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            f_min=config.f_min,
            f_max=config.f_max,
        )
        stats = [-5.4919195,  5.0389895] # provided by authors
        self.normalizer = PrecomputedNorm(stats)

        # Load pretrained weights.
        self.model = AudioNTT2020(d=config.feature_d)
        self.model.load_weight(ckpt, device='cpu')

        # attributes
        self.output_dim = config.feature_d
        self.max_input_length = config.shape[-1]

    # Interface
    def get_output_dim(self):
        return self.output_dim

    # Interface
    def get_downsample_rates(self, key: str) -> int:
        return 15344.655344655344 # computed by: len(wavs[0]) / len(features[0]) * self.max_input_length

    # forward in chunks
    def forward_in_chunks(self, features):
        outputs = []
        for i in range(0, features.size(1), self.max_input_length):
            subseq = features[:, i:i+self.max_input_length, :]
            if subseq.size(1) < self.max_input_length: break # break if the chunk is too small for the model to forward
            feats = self.model(subseq.permute(0, 2, 1).unsqueeze(1)) # features: (B, 1, F, T)
            outputs.append(feats.unsqueeze(1)) # (B, 1, D)
        outputs = torch.cat(outputs, dim=1) # (B, T, D)
        return outputs
    
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
        features = [self.normalizer((self.to_melspec(wav) + torch.finfo(torch.float).eps).log()).permute(1, 0) for wav in wavs] # features: (B, T, F)
        features = pad_sequence(features, batch_first=True)

        # forward the sequence in chunks then concat
        features = self.forward_in_chunks(features)
        return {
            "last_hidden_state": features,
            "hidden_states": [features],
        }
