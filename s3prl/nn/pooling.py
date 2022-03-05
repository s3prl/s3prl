import torch

from . import NNModule


class MeanPooling(NNModule):
    def __init__(self):
        super().__init__()

    @property
    def input_size(self):
        raise ValueError

    @property
    def output_size(self):
        raise ValueError

    def forward(self, xs, xs_len=None):
        pooled_list = []
        for x, x_len in zip(xs, xs_len):
            pooled = torch.mean(x[:x_len], dim=0)
            pooled_list.append(pooled)
        return torch.stack(pooled_list)
