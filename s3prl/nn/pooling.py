import torch
import torch.nn as nn
import torch.nn.functional as F

from s3prl import Output

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


# TODO: add x_len into pooling
class TemporalAveragePooling(NNModule):
    def __init__(self, input_size: int, output_size: int):
        """
        TemporalAveragePooling
        Paper: Multi-Task Learning with High-Order Statistics for X-vector based Text-Independent Speaker Verification
        Link: https://arxiv.org/pdf/1903.12058.pdf
        """
        super().__init__()

    @property
    def input_size(self):
        return self.arguments.input_size

    @property
    def output_size(self):
        return self.arguments.output_size

    def forward(self, xs, xs_len=None):
        """
        Computes Temporal Average Pooling Module
        Args:
            xs (torch.Tensor): Input tensor (#batch, channels, frames).
            xs_len: with the lengths for each sample
        Returns:
            torch.Tensor: Output tensor (#batch, channels)
        """
        pooled_list = []
        for x, x_len in zip(xs, xs_len):
            pooled = torch.mean(x[:x_len], dim=0)
            pooled_list.append(pooled)

        return torch.stack(pooled_list)


class TemporalStatisticsPooling(NNModule):
    def __init__(self, input_size: int, output_size: int):
        """
        TemporalStatisticsPooling
        Paper: X-vectors: Robust DNN Embeddings for Speaker Recognition
        Link： http://www.danielpovey.com/files/2018_icassp_xvectors.pdf
        """
        super().__init__()

    @property
    def input_size(self):
        return self.arguments.input_size

    @property
    def output_size(self):
        return self.arguments.output_size

    def forward(self, xs, xs_len=None):
        """
        Computes Temporal Statistics Pooling Module
        Args:
            xs (torch.Tensor): Input tensor (#batch, channels, frames).
            xs_len: with the lengths for each sample
        Returns:
            torch.Tensor: Output tensor (#batch, channels)
        """
        pooled_list = []
        for x, x_len in zip(xs, xs_len):
            mean = torch.mean(x[:x_len], dim=0)
            var = torch.var(x[:x_len], dim=0)
            pooled = torch.cat((mean, var))
            pooled_list.append(pooled)
        return torch.stack(pooled_list)


class SelfAttentivePooling(NNModule):
    def __init__(self, input_size: int, output_size: int):
        """
        SelfAttentivePooling
        Paper: Self-Attentive Speaker Embeddings for Text-Independent Speaker Verification
        Link： https://danielpovey.com/files/2018_interspeech_xvector_attention.pdf
        """
        super().__init__()
        self.sap_linear = nn.Linear(input_size, input_size)
        self.attention = nn.Parameter(torch.FloatTensor(input_size, 1))

    @property
    def input_size(self):
        return self.arguments.input_size

    @property
    def output_size(self):
        return self.arguments.output_size

    def forward(self, xs, xs_len=None):
        """
        Computes Self-Attentive Pooling Module
        Args:
            xs (torch.Tensor): Input tensor (#batch, channels, frames).
            xs_len: with the lengths for each sample
        Returns:
            torch.Tensor: Output tensor (#batch, channels)
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


class AttentiveStatisticsPooling(NNModule):
    def __init__(self, input_size: int, output_size: int):
        """
        AttentiveStatisticsPooling
        Paper: Attentive Statistics Pooling for Deep Speaker Embedding
        Link: https://arxiv.org/pdf/1803.10963.pdf
        """
        super().__init__()
        self.sap_linear = nn.Linear(input_size, input_size)
        self.attention = nn.Parameter(torch.FloatTensor(input_size, 1))

    @property
    def input_size(self):
        return self.arguments.input_size

    @property
    def output_size(self):
        return self.arguments.output_size

    def forward(self, xs, xs_len=None):
        """
        Computes Attentive Statistics Pooling Module
        Args:
            xs (torch.Tensor): Input tensor (#batch, channels, frames).
            xs_len: with the lengths for each sample
        Returns:
            torch.Tensor: Output tensor (#batch, channels)
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
