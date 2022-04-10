from typing import List

import torch
import torch.nn as nn
from s3prl import Output

from . import NNModule


def downsample(x, x_len, sample_rate: int, sample_style: str):
    raise NotImplementedError


class RNNLayer(NNModule):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        module: str,
        dropout: float = 0.0,
        bidirectional: bool = False,
        proj: bool = False,
        layer_norm: bool = False,
        sample_rate: int = 1,
    ):
        super().__init__()

    def forward(self, xs, xs_len) -> Output:
        raise NotImplementedError

    @property
    def input_size(self):
        return self.arguments.input_size

    @property
    def output_size(self):
        raise self.arguments.hidden_size


class RNNEncoder(NNModule):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        upstream_rate: int,
        module: str,
        hidden_size: List[int],
        dropout: List[float],
        layer_norm: List[bool],
        proj: List[bool],
        sample_rate: List[int],
        sample_style: str,
        bidirectional: bool = False,
        total_rate: int = 320,
    ):
        super().__init__()

    def forward(self, xs, xs_len) -> Output:
        raise NotImplementedError

    @property
    def input_size(self):
        return self.arguments.input_size

    @property
    def output_size(self):
        raise self.arguments.output_size
