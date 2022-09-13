from typing import List

import torch.nn as nn

from s3prl.nn.pooling import MeanPooling

__all__ = [
    "FrameLevel",
    "UtteranceLevel",
]


class FrameLevel(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: List[int] = [256],
        activation_cls: type = None,
    ):
        super().__init__()
        self._indim = input_size
        self._outdim = output_size

        hidden_sizes = hidden_sizes
        latest_size = input_size

        hidden_layers = []
        if len(hidden_sizes) > 0:
            for size in hidden_sizes:
                hidden_layers.append(nn.Linear(latest_size, size))
                if activation_cls is not None:
                    hidden_layers.append(activation_cls())
                latest_size = size

        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.final_proj = nn.Linear(latest_size, output_size)

    @property
    def input_size(self):
        return self._indim

    @property
    def output_size(self):
        return self._outdim

    def forward(self, x, x_len):
        ys = self.hidden_layers(x)
        ys = self.final_proj(ys)
        return ys, x_len


class UtteranceLevel(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: List[int] = [256],
        activation: str = None,
        pooling_cls: type = MeanPooling,
        pooling_conf: dict = None,
    ):
        super().__init__()
        self._indim = input_size
        self._outdim = output_size
        latest_size = input_size

        hidden_layers = []
        if len(hidden_sizes) > 0:
            for size in hidden_sizes:
                hidden_layers.append(nn.Linear(latest_size, size))
                if activation is not None:
                    hidden_layers.append(getattr(nn, activation)())
                latest_size = size

        self.hidden_layers = nn.Sequential(*hidden_layers)

        pooling_conf = pooling_conf or {}
        self.pooling = pooling_cls(**pooling_conf)
        self.final_proj = nn.Linear(latest_size, output_size)

    @property
    def input_size(self):
        return self._indim

    @property
    def output_size(self):
        return self._outdim

    def forward(self, x, x_len):
        x = self.hidden_layers(x)
        x_pooled = self.pooling(x, x_len)
        y = self.final_proj(x_pooled)
        return y
