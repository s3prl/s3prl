# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/decoar/expert.py ]
#   Synopsis     [ the decoar wrapper ]
#   Author       [ awslabs/speech-representations ]
#   Reference    [ https://github.com/awslabs/speech-representations ]
"""*********************************************************************************************"""


import torch
import torchaudio
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from upstream.interfaces import UpstreamBase
import mxnet as mx
from speech_reps.featurize import DeCoARFeaturizer
from utility.helper import show

SAMPLE_RATE = 16000
EXAMPLE_SEC = 5
DECOAR_NUM_MEL_BINS = 40


class UpstreamExpert(UpstreamBase):
    def __init__(self, ckpt, **kwargs):
        super().__init__(**kwargs)

        _load_wav = torchaudio.load_wav

        def load_short_and_turn_float(*args, **kwargs):
            wav, sr = _load_wav(*args, **kwargs)
            if wav.dtype is not torch.short:
                show(
                    f"[Warning] - Decoar only takes .wav files for the official usage."
                )
                show(f"[Warning] - {args[0]} is not a .wav file")
            wav = wav.float()
            return wav, sr

        setattr(torchaudio, "load", load_short_and_turn_float)

        # Decoar can not easily switch between cpu/gpu for now
        self.model = DeCoARWavFeaturizer(ckpt, gpu=0)

    def forward(self, wavs):
        feature = self.model.wav_to_feats(wavs)
        return {
            "last_hidden_state": feature,
            "hidden_states": [feature],
        }


class DeCoARWavFeaturizer(DeCoARFeaturizer):
    def __init__(self, params_file, gpu=None, eps=1e-20):
        super().__init__(params_file, gpu=gpu)
        self.eps = eps

    def wav_to_feats(self, wavs):
        device = wavs[0].device

        def extract_fbank_cmvn(wav):
            fbank = torchaudio.compliance.kaldi.fbank(
                wav.unsqueeze(0), num_mel_bins=DECOAR_NUM_MEL_BINS
            )
            cmvn = (fbank - fbank.mean(dim=0, keepdim=True)) / (
                fbank.std(dim=0, keepdim=True) + self.eps
            )
            return cmvn

        raw_feats = [extract_fbank_cmvn(wav).cpu() for wav in wavs]
        raw_feats_len = torch.LongTensor([raw_feat.size(0) for raw_feat in raw_feats])
        raw_feats_padded = pad_sequence(raw_feats)

        data = mx.nd.array(raw_feats_padded.detach().numpy(), ctx=self._ctx)
        data_len = mx.nd.array(raw_feats_len.detach().numpy(), ctx=self._ctx)

        vecs = self._model(data, data_len)
        reps = (
            torch.FloatTensor(vecs.asnumpy()).transpose(0, 1).contiguous()
        )  # B * T * C
        reps = reps.to(device=device)

        return reps
