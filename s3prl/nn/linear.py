from typing import List

import torch
import torch.nn as nn

from s3prl import Output

from . import NNModule
from .pooling import MeanPooling


class FrameLevelLinear(NNModule):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: List[int] = None,
        **kwds,
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

    def forward(self, x, x_len):
        ys = self.hidden_layers(x)
        ys = self.model(ys)
        return Output(output=ys, output_len=x_len)


class MeanPoolingLinear(NNModule):
    def __init__(
        self, input_size: int, output_size: int, hidden_size: int = 256, **kwargs
    ):
        super().__init__()
        self.pre_linear = nn.Linear(input_size, hidden_size)
        self.pooling = MeanPooling()
        self.post_linear = nn.Linear(hidden_size, output_size)

    @property
    def input_size(self):
        return self.arguments.input_size

    @property
    def output_size(self):
        return self.arguments.output_size

    def forward(self, xs, xs_len=None):
        xs = self.pre_linear(xs)
        xs_pooled = self.pooling(xs, xs_len)
        ys = self.post_linear(xs_pooled)
        return Output(output=ys)
