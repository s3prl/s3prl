import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio import transforms
from collections import namedtuple

SAMPLE_RATE = 16000
EXAMPLE_SEC = 5


def get_extracter(config):
    transforms = [
        ExtractAudioFeature(**config.get('kaldi', {})),
        Delta(**config.get('delta', {})),
        CMVN(**config.get('cmvn', {})),
    ]
    extracter = nn.Sequential(*transforms)
    output_dim = extracter(torch.randn(EXAMPLE_SEC * SAMPLE_RATE)).size(-1)

    return extracter, output_dim


class ExtractAudioFeature(nn.Module):
    def __init__(self, feat_type='fbank', **kwargs):
        super(ExtractAudioFeature, self).__init__()
        self.extract_fn = eval(f'torchaudio.compliance.kaldi.{feat_type}')
        self.kwargs = kwargs[feat_type]

    def forward(self, waveform):
        # waveform: (time, )
        x = self.extract_fn(
            waveform.view(1, -1),
            sample_frequency = SAMPLE_RATE,
            **self.kwargs
        )
        # x: (feat_seqlen, feat_dim)
        return x


class Delta(nn.Module):
    def __init__(self, order=2, **kwargs):
        super(Delta, self).__init__()
        self.order = order
        self.compute_delta = transforms.ComputeDeltas(**kwargs)

    def forward(self, x):
        # x: (feat_seqlen, feat_dim)
        feats = [x]
        for o in range(self.order):
            feat = feats[-1].transpose(0, 1).unsqueeze(0)
            delta = self.compute_delta(feat)
            feats.append(delta.squeeze(0).transpose(0, 1))
        x = torch.cat(feats, dim=-1)
        # x: (feat_seqlen, feat_dim)
        return x


class CMVN(nn.Module):
    def __init__(self, use_cmvn, eps=1e-10):
        super(CMVN, self).__init__()
        self.eps = eps
        self.use_cmvn = use_cmvn

    def forward(self, x):
        # x: (feat_seqlen, feat_dim)
        if self.use_cmvn:
            x = (x - x.mean(dim=0, keepdim=True)) / (self.eps + x.std(dim=0, keepdim=True))
        return x

