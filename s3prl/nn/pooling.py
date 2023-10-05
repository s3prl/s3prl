"""
Common pooling methods

Authors:
  * Leo 2022
  * Haibin Wu 2022
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "MeanPooling",
    "TemporalAveragePooling",
    "TemporalStatisticsPooling",
    "SelfAttentivePooling",
    "AttentiveStatisticsPooling",
]


class MeanPooling(nn.Module):
    """
    Computes Temporal Average Pooling (MeanPooling over time) Module
    """

    def __init__(self, input_size: int):
        super().__init__()
        self._in_size = input_size

    @property
    def input_size(self) -> int:
        return self._in_size

    @property
    def output_size(self) -> int:
        return self._in_size

    def forward(self, xs: torch.Tensor, xs_len: torch.LongTensor):
        """
        Args:
            xs (torch.Tensor): Input tensor (#batch, frames, input_size).
            xs_len (torch.LongTensor): with the lengths for each sample
        Returns:
            torch.Tensor: Output tensor (#batch, input_size)
        """
        pooled_list = []
        for x, x_len in zip(xs, xs_len):
            pooled = torch.mean(x[:x_len], dim=0)
            pooled_list.append(pooled)
        return torch.stack(pooled_list)


TemporalAveragePooling = MeanPooling


class TemporalStatisticsPooling(nn.Module):
    """
    TemporalStatisticsPooling
    Paper: X-vectors: Robust DNN Embeddings for Speaker Recognition
    Link: http://www.danielpovey.com/files/2018_icassp_xvectors.pdf
    """

    def __init__(self, input_size: int):
        super().__init__()
        self._input_size = input_size

    @property
    def input_size(self) -> int:
        return self._input_size

    @property
    def output_size(self) -> int:
        return self._input_size * 2

    def forward(self, xs, xs_len):
        """
        Computes Temporal Statistics Pooling Module

        Args:
            xs (torch.Tensor): Input tensor (#batch, frames, input_size).
            xs_len (torch.LongTensor): with the lengths for each sample

        Returns:
            torch.Tensor: Output tensor (#batch, output_size)
        """
        pooled_list = []
        for x, x_len in zip(xs, xs_len):
            mean = torch.mean(x[:x_len], dim=0)
            std = torch.std(x[:x_len], dim=0)
            pooled = torch.cat((mean, std), dim=-1)
            pooled_list.append(pooled)
        return torch.stack(pooled_list)


class SelfAttentivePooling(nn.Module):
    """
    SelfAttentivePooling
    Paper: Self-Attentive Speaker Embeddings for Text-Independent Speaker Verification
    Link: https://danielpovey.com/files/2018_interspeech_xvector_attention.pdf
    """

    def __init__(self, input_size: int):
        super().__init__()
        self._indim = input_size
        self.sap_linear = nn.Linear(input_size, input_size)
        self.attention = nn.Parameter(torch.FloatTensor(input_size, 1))

    @property
    def input_size(self) -> int:
        return self._indim

    @property
    def output_size(self) -> int:
        return self._indim

    def forward(self, xs, xs_len):
        """
        Computes Self-Attentive Pooling Module

        Args:
            xs (torch.Tensor): Input tensor (#batch, frames, input_size).
            xs_len (torch.LongTensor): with the lengths for each sample

        Returns:
            torch.Tensor: Output tensor (#batch, input_size)
        """
        pooled_list = []
        for x, x_len in zip(xs, xs_len):
            x = x[:x_len].unsqueeze(0)
            h = torch.tanh(self.sap_linear(x))
            w = torch.matmul(h, self.attention).squeeze(dim=2)
            w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1)
            x = torch.sum(x * w, dim=1)
            pooled_list.append(x.squeeze(0))
        return torch.stack(pooled_list)


class AttentiveStatisticsPooling(nn.Module):
    """
    AttentiveStatisticsPooling
    Paper: Attentive Statistics Pooling for Deep Speaker Embedding
    Link: https://arxiv.org/pdf/1803.10963.pdf
    """

    def __init__(self, input_size: int):
        super().__init__()
        self._indim = input_size
        self.sap_linear = nn.Linear(input_size, input_size)
        self.attention = nn.Parameter(torch.FloatTensor(input_size, 1))

    @property
    def input_size(self) -> int:
        return self._indim

    @property
    def output_size(self) -> int:
        return self._indim * 2

    def forward(self, xs, xs_len):
        """
        Computes Attentive Statistics Pooling Module

        Args:
            xs (torch.Tensor): Input tensor (#batch, frames, input_size).
            xs_len (torch.LongTensor): with the lengths for each sample

        Returns:
            torch.Tensor: Output tensor (#batch, input_size)
        """
        pooled_list = []
        for x, x_len in zip(xs, xs_len):
            x = x[:x_len].unsqueeze(0)
            h = torch.tanh(self.sap_linear(x))
            w = torch.matmul(h, self.attention).squeeze(dim=2)
            w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1)
            mu = torch.sum(x * w, dim=1)
            rh = torch.sqrt((torch.sum((x**2) * w, dim=1) - mu**2).clamp(min=1e-5))
            x = torch.cat((mu, rh), 1).squeeze(0)
            pooled_list.append(x)
        return torch.stack(pooled_list)
