import abc

import torch
import torch.nn as nn


class UpstreamExpertInterface(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, eps=1.0e-12, *args, **kwargs):
        super(UpstreamExpertInterface, self).__init__()
        self.eps = eps

    @classmethod
    def __subclasshook__(cls, subclass):
        checklist = [
            "output_dim",
            "downsample_rate",
            "forward"
        ]
        for checkitem in checklist:
            if not hasattr(subclass, checkitem):
                return NotImplemented
        return True

    @property
    @abc.abstractmethod
    def output_dim(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def downsample_rate(self):
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, wavs: [torch.FloatTensor]):
        raise NotImplementedError

    def normalize(self, wavs: [torch.FloatTensor]):
        wavs = [(x - x.mean()) / (x.std() + self.eps) for x in wavs]
        return wavs
