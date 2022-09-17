# --------------------------------------------------------
# LightHuBERT: Lightweight and Configurable Speech Representation Learning with Once-for-All Hidden-Unit BERT (https://arxiv.org/pdf/2203.15610.pdf)
# Github source: https://github.com/mechanicalsea/lighthubert
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SLinear(nn.Linear):
    r"""Linear Layer: variable input and output size.
    `in_splits` and `out_splits` enable equally seperate splits
        at in and out dimensions, e.g.,
        A tensor of 128 size has 4 splits with the split [0,31],
        the split [32,63], the split [64,95], and the split
        [96,127].
        If given 2.5 splits and total 40 sizes, we can conclude
        that the split [0-15], the split [32-47], and the split
        [64-71].

    __base__: torch.nn.Linear
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        in_splits: int = 1,
        out_splits: int = 1,
    ) -> None:
        super(SLinear, self).__init__(in_features, out_features, bias)
        self.in_splits = in_splits
        self.out_splits = out_splits
        assert self.in_features % self.in_splits == 0
        assert self.out_features % self.out_splits == 0
        self.in_size_split = self.in_features // self.in_splits
        self.out_size_split = self.out_features // self.out_splits
        self.staticize()

    def staticize(self):
        self.sample_in_dim = None
        self.sample_out_dim = None
        self.sample_in_splits = None
        self.sample_out_splits = None
        self.samples = {
            "weight": self.weight,
            "bias": self.bias,
        }

    def set_sample_config(
        self, sample_in_dim, sample_out_dim, sample_in_splits=1, sample_out_splits=1
    ):
        assert sample_in_dim is not None and sample_out_dim is not None
        if sample_in_splits is not None:
            assert sample_in_splits <= self.in_splits, f"{sample_in_splits}"
            assert (
                sample_in_dim // sample_in_splits > 0
            ), f"{sample_in_dim} // {sample_in_splits} <= 0"
        if sample_out_splits is not None:
            assert sample_out_splits <= self.out_splits, f"{sample_out_splits}"
            assert (
                sample_out_dim // sample_out_splits > 0
            ), f"{sample_out_dim} // {sample_out_splits} <= 0"
        self.sample_in_dim = sample_in_dim
        self.sample_out_dim = sample_out_dim
        self.sample_in_splits = sample_in_splits
        self.sample_out_splits = sample_out_splits
        self._sample_parameters()

    def _sample_parameters(self):
        if (
            self.in_splits == 1
            or self.sample_in_dim == self.in_features
            or self.sample_in_splits is None
        ):
            weight = self.weight[:, : self.sample_in_dim].contiguous()
        else:
            assert self.sample_in_splits is not None, f"{self.sample_in_splits}"
            weight_splits = torch.split(self.weight, self.in_size_split, dim=1)
            size_split = int(self.sample_in_dim / self.sample_in_splits)
            weight = torch.cat(
                [
                    split_i[:, : min(size_split, self.sample_in_dim - i * size_split)]
                    for i, split_i in enumerate(
                        weight_splits[: int(np.ceil(self.sample_in_splits))]
                    )
                ],
                dim=1,
            )

        if (
            self.out_splits == 1
            or self.sample_out_dim == self.out_features
            or self.sample_out_splits is None
        ):
            weight = weight[: self.sample_out_dim].contiguous()
            if self.bias is not None:
                bias = self.bias[: self.sample_out_dim].contiguous()
        else:
            assert self.sample_out_splits is not None, f"{self.sample_out_splits}"
            weight_splits = torch.split(weight, self.out_size_split, dim=0)
            size_split = int(self.sample_out_dim / self.sample_out_splits)
            weight = torch.cat(
                [
                    split_i[: min(size_split, self.sample_out_dim - i * size_split)]
                    for i, split_i in enumerate(
                        weight_splits[: int(np.ceil(self.sample_out_splits))]
                    )
                ],
                dim=0,
            )
            if self.bias is not None:
                bias_splits = torch.split(self.bias, self.out_size_split, dim=0)
                bias = torch.cat(
                    [
                        split_i[: min(size_split, self.sample_out_dim - i * size_split)]
                        for i, split_i in enumerate(
                            bias_splits[: int(np.ceil(self.sample_out_splits))]
                        )
                    ],
                    dim=0,
                )

        self.samples["weight"] = weight
        if self.bias is not None:
            self.samples["bias"] = bias

        # self.samples["weight"] = self.weight[:self.sample_out_dim, :self.sample_in_dim]
        # if self.bias is not None:
        #     self.samples["bias"] = self.bias[:self.sample_out_dim]
        return self.samples

    def calc_sampled_param_num(self):
        weight_numel = self.samples["weight"].numel()

        if self.samples["bias"] is not None:
            bias_numel = self.samples["bias"].numel()
        else:
            bias_numel = 0

        return weight_numel + bias_numel

    def get_complexity(self, sequence_length):
        total_flops = 0
        total_flops += sequence_length * np.prod(self.samples["weight"].size())
        return total_flops

    @property
    def weights(self):
        return self.samples["weight"]

    @property
    def biases(self):
        return self.samples["bias"]

    def forward(self, input: Tensor) -> Tensor:
        self._sample_parameters()
        return F.linear(input, self.weights, self.biases)

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )

    def clone_model(
        self, in_dim: int, out_dim: int, in_splits: int = 1, out_splits: int = 1
    ):
        self.set_sample_config(in_dim, out_dim, in_splits, out_splits)

        isbias = self.bias is not None
        m = nn.Linear(in_dim, out_dim, isbias)
        m = m.to(self.weight.device)
        m = m.to(self.weight.dtype)
        m.weight.data.copy_(self.weights)
        if isbias:
            m.bias.data.copy_(self.biases)
        return m.eval()

    @classmethod
    def build_from(cls, m: nn.Linear, in_splits: int = 1, out_splits: int = 1):
        in_features = m.in_features
        out_features = m.out_features
        isbias = m.bias is not None
        _m = cls(in_features, out_features, isbias, in_splits, out_splits)
        _m = _m.to(m.weight.device)
        _m = _m.to(m.weight.dtype)
        _m.weight.data.copy_(m.weight)
        if isbias:
            _m.bias.data.copy_(m.bias)
        return _m


if __name__ == "__main__":
    print(SLinear.__base__)
    m = SLinear(2, 3)
    z = m.clone_model(2, 2)
    x = SLinear.build_from(z)
    print(m, z, x)
    inp = torch.rand((1, 2))
    m.set_sample_config(2, 2)
    print(torch.allclose(m(inp), z(inp)))
    print(torch.allclose(m(inp), x(inp)))

    m = SLinear(2, 4, in_splits=2, out_splits=4)
    z = m.clone_model(2, 2, 2, 2)
    x = SLinear.build_from(z, 2, 2)
    inp = torch.rand((1, 2))
    m.set_sample_config(2, 2, 2, 2)
    print(torch.allclose(m(inp), z(inp)))
    print(torch.allclose(m(inp), x(inp)))
