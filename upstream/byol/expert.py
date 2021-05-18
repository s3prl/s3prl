# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/byol/expert.py ]
#   Synopsis     [ the byol wrapper ]
#   Author       [ Andy T. Liu (https://github.com/andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import math
import yaml
#-------------#
import torch
from torch.nn.utils.rnn import pad_sequence
#-------------#
from upstream.baseline.preprocessor import get_preprocessor
from .model import AudioEncoder


###################
# UPSTREAM EXPERT #
###################
class UpstreamExpert(torch.nn.Module):
    """
    The Mockingjay wrapper
    """

    def __init__(self, ckpt, feature_selection=-1, model_config=None, **kwargs):
        super(UpstreamExpert, self).__init__()

        if model_config is not None:
            raise NotImplementedError

        all_states = torch.load(ckpt, map_location='cpu')
        self.upstream_config = all_states['Config']
        self.target_level = self.upstream_config['audio']['target_level']
        self.max_input_length = self.upstream_config['task']['sequence_length']

        self.extracter, input_dim, _ = get_preprocessor(self.upstream_config['audio'])
        self.model = AudioEncoder(input_dim, **self.upstream_config['audio_encoder'])
        #self.model.load_state_dict(all_states['Model'])

    # Interface
    def get_output_dim(self):
        return self.upstream_config['audio_encoder']['dim']

    # Interface
    def get_downsample_rate(self):
        return 160

    def _normalize_wav_decibel(self, wav):
        '''Normalize the signal to the target level'''
        rms = wav.pow(2).mean().pow(0.5)
        scalar = (10 ** (self.target_level / 20)) / (rms + 1e-10)
        wav = wav * scalar
        return wav

    def preprocess(self, wavs):
        norm_wavs = [self._normalize_wav_decibel(wav) for wav in wavs]
        padd_wavs = pad_sequence(norm_wavs, batch_first=True)
        padd_wavs = padd_wavs.unsqueeze(1) # (batch_size, audio_len) -> (batch_size, 1, audio_len)
        feats = self.extracter(padd_wavs)[0]
        return feats

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

        feats = self.preprocess(wavs)

        # forward the sequence in chunks then concat
        chunks = torch.chunk(feats, chunks=math.ceil(feats.size(1) / self.max_input_length), dim=1)
        features = []
        for chunk in chunks:
            outputs = self.model(chunk, enc_len=None)
            features.append(outputs[1]) # content_input_x
        features = torch.cat(features, dim=1) # (batch_size, extracted_seqlen, feature_dim)

        ratio = len(features[0]) / wav_lengths[0]
        feat_lengths = [round(l * ratio) for l in wav_lengths]
        features = [f[:l] for f, l in zip(features, feat_lengths)]

        return features