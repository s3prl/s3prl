# --------------------------------------------------------
# LightHuBERT: Lightweight and Configurable Speech Representation Learning with Once-for-All Hidden-Unit BERT (https://arxiv.org/pdf/2203.15610.pdf)
# Github source: https://github.com/mechanicalsea/lighthubert
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import torch
import torch.nn as nn


class SConv1d(nn.Conv1d):
    """SCon1d (Scaling Conv1d): support variable in channels and out channels.

    Notes
    -----
        WeightNorm has `weight_v` and `weight_g`, where
            weight = weight_g / ||weight_v|| * weight_v
        weight is obtained before forward.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ):
        """Add dynamic hyper-parameters"""
        super(SConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.staticize()

    def staticize(self):
        self.sample_in_channels = None
        self.sample_out_channels = None

    def set_sample_config(self, sample_in_channels, sample_out_channels):
        """Sampling subnet"""
        self.sample_in_channels = sample_in_channels
        self.sample_out_channels = sample_out_channels

    def _sample_parameters(self, weight, bias=None):
        """Use to method: set_sample_config"""
        out_dim = self.sample_out_channels
        in_dim = self.sample_in_channels // self.groups
        weight = weight[:out_dim, :in_dim, :]
        if bias is not None:
            bias = bias[:out_dim]
        return weight, bias

    def forward(self, x):
        weight, bias = self._sample_parameters(self.weight, self.bias)
        return self._conv_forward(x, weight, bias)

    def calc_sampled_param_num(self):
        """Calculating the parameters of a subnet"""
        if self.sample_in_channels is not None:
            weight_numel = self.sample_in_channels * self.sample_out_channels
        else:
            weight_numel = self.in_features * self.out_features
        weight_numel = weight_numel * self.kernel_size[0] // self.groups

        if self.bias is not None:
            if self.sample_in_channels is not None:
                bias_numel = self.sample_out_channels
            else:
                bias_numel = self.out_features
        else:
            bias_numel = 0

        return weight_numel + bias_numel

    def get_complexity(self, sequence_length):
        """Getting computational complexity.
        ref: https://github.com/Lyken17/pytorch-OpCounter/blob/509fb7e7e48faaddd4b436371ccd39ede77f1f4a/thop/vision/counter.py#L16
        """
        total_flops = 0
        output_size = sequence_length * self.samples["weight"].size(0)
        in_channel_div_groups = self.samples["weight"].size(1)
        kernel_size = self.kernel_size
        bias = 1 if self.samples["bias"] is not None else 0
        total_flops += output_size * (in_channel_div_groups * kernel_size + bias)
        return total_flops

    def clone_model(self, in_channels: int, out_channels: int):
        """Clone a subnet instance with its supernet's weights."""
        self.set_sample_config(in_channels, out_channels)
        isbias = self.bias is not None
        m = nn.Conv1d(
            in_channels,
            out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            isbias,
            self.padding_mode,
        )
        m = m.to(self.weight.device)
        m = m.to(self.weight.dtype)
        weight, bias = self._sample_parameters(self.weight, self.bias)
        m.weight.data.copy_(weight)
        if isbias:
            m.bias.data.copy_(bias)
        return m.eval()

    @classmethod
    def build_from(cls, m: nn.Conv1d):
        isbias = m.bias is not None
        _m = cls(
            m.in_channels,
            m.out_channels,
            m.kernel_size,
            m.stride,
            m.padding,
            m.dilation,
            m.groups,
            isbias,
            m.padding_mode,
        )
        _m = _m.to(m.weight.device)
        _m = _m.to(m.weight.dtype)
        _m.weight.data.copy_(m.weight)
        if isbias:
            _m.bias.data.copy_(m.bias)
        return _m


if __name__ == "__main__":
    print(SConv1d.__base__)
    m = SConv1d(4, 4, 2, padding=2, groups=2)
    z = m.clone_model(2, 2)
    x = SConv1d.build_from(z)
    x.set_sample_config(2, 2)
    print(m, z, x)
    inp = torch.rand((1, 2, 3))
    m.set_sample_config(2, 2)
    print(torch.allclose(m(inp), z(inp)))
    print(torch.allclose(m(inp), x(inp)))
    m.set_sample_config(2, 1)
    print(type(m.weight))
    print(m.weight.dtype)
    print(m.weight.shape)
    print(m.weight)
