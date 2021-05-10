# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/decoar/expert.py ]
#   Synopsis     [ the decoar wrapper ]
#   Author       [ awslabs/speech-representations ]
#   Reference    [ https://github.com/awslabs/speech-representations ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import torch
import torchaudio
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
#-------------#
import mxnet as mx
from speech_reps.featurize import DeCoARFeaturizer
from utility.helper import show


############
# CONSTANT #
############
SAMPLE_RATE = 16000
EXAMPLE_SEC = 5
DECOAR_NUM_MEL_BINS = 40


###################
# UPSTREAM EXPERT #
###################
class UpstreamExpert(nn.Module):
    """
    The DeCoAR wrapper
    """

    def __init__(self, ckpt, **kwargs):
        super(UpstreamExpert, self).__init__() 

        _load_wav = torchaudio.load_wav
        def load_short_and_turn_float(*args, **kwargs):
            wav, sr = _load_wav(*args, **kwargs)
            if wav.dtype is not torch.short:
                show(f'[Warning] - Decoar only takes .wav files for the official usage.')
                show(f'[Warning] - {args[0]} is not a .wav file')
            wav = wav.float()
            return wav, sr
        setattr(torchaudio, 'load', load_short_and_turn_float)

        # Decoar can not easily switch between cpu/gpu for now
        self.model = DeCoARWavFeaturizer(ckpt, gpu=0)
        pseudo_input = torch.randn(SAMPLE_RATE * EXAMPLE_SEC).cuda()
        self.output_dim = self.model.wav_to_feats([pseudo_input])[0].size(-1)

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
        return self.model.wav_to_feats(wavs)


class DeCoARWavFeaturizer(DeCoARFeaturizer):
    def __init__(self, params_file, gpu=None, eps=1e-20):
        super().__init__(params_file, gpu=gpu)
        self.eps = eps

    def wav_to_feats(self, wavs):
        """
        Args:
            wavs: a list of wavs in torch.FloatTensor which are loaded without normalization
        """
        device = wavs[0].device

        def extract_fbank_cmvn(wav):
            fbank = torchaudio.compliance.kaldi.fbank(wav.unsqueeze(0), num_mel_bins=DECOAR_NUM_MEL_BINS)
            cmvn = (fbank - fbank.mean(dim=0, keepdim=True)) / (fbank.std(dim=0, keepdim=True) + self.eps)
            return cmvn

        raw_feats = [extract_fbank_cmvn(wav).cpu() for wav in wavs]
        raw_feats_len = torch.LongTensor([raw_feat.size(0) for raw_feat in raw_feats])
        raw_feats_padded = pad_sequence(raw_feats)

        data = mx.nd.array(raw_feats_padded.detach().numpy(), ctx=self._ctx)
        data_len = mx.nd.array(raw_feats_len.detach().numpy(), ctx=self._ctx)

        vecs = self._model(data, data_len)
        reps = torch.FloatTensor(vecs.asnumpy()).transpose(0, 1).contiguous() # B * T * C
        reps = reps.to(device=device)

        return [rep[:l] for rep, l in zip(reps, raw_feats_len)]
