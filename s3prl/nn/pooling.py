import torch
import torch.nn as nn

from s3prl import init
from . import Module


class MeanPooling(Module):
    @init.method
    def __init__(self):
        super().__init__()

    def forward(self, xs, xs_len=None):
        pooled_list = []
        for x, x_len in zip(xs, xs_len):
            pooled = torch.mean(x[:x_len], dim=0)
            pooled_list.append(pooled)
        return torch.stack(pooled_list)
