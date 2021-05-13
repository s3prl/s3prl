import abc

import torch
import torch.nn as nn


class UpstreamExpertInterface(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, normalize_wav=False, eps=1.0e-12, **kwargs):
        super(UpstreamExpertInterface, self).__init__()
        self.normalize_wav = normalize_wav
        self.eps = eps

    @abc.abstractmethod
    def get_output_dim(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_downsample_rate(self):
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, wavs: [torch.FloatTensor]):
        raise NotImplementedError

    def preprocess(self, wavs: [torch.FloatTensor]):
        if self.normalize_wav:
            wavs = [(x - x.mean()) / (x.std() + self.eps) for x in wavs]
        return wavs

    def postprocess(self, features: [torch.FloatTensor]):
        return features
