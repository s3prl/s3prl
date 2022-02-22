from typing import List

import torch
import torch.nn as nn
from s3prl import init
from s3prl.base.output import Output

from . import Module
from .pooling import MeanPooling


class FrameLevelLinear(Module):
    @init.method
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: List[int] = None,
    ):
        super().__init__()
        latest_size = input_size
        hidden_layers = []
        if hidden_sizes is not None:
            for size in hidden_sizes:
                hidden_layers += [
                    nn.Linear(latest_size, size),
                ]
                latest_size = size
        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.model = nn.Linear(latest_size, output_size)

    @property
    def input_size(self):
        return self.arguments.input_size

    @property
    def output_size(self):
        return self.arguments.output_size

    def forward(self, xs):
        ys = self.hidden_layers(xs)
        ys = self.model(ys)
        return ys


class MeanPoolingLinear(Module):
    @init.method
    def __init__(self, input_size: int, output_size: int, **kwargs):
        super().__init__()
        self.pooling = MeanPooling()
        self.linear = FrameLevelLinear(input_size, output_size)

    def forward(self, xs, xs_len=None):
        xs_pooled = self.pooling(xs, xs_len)
        ys = self.linear(xs_pooled)
        return ys
