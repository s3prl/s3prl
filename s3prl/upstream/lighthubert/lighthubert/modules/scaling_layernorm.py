# --------------------------------------------------------
# LightHuBERT: Lightweight and Configurable Speech Representation Learning with Once-for-All Hidden-Unit BERT (https://arxiv.org/pdf/2203.15610.pdf)
# Github source: https://github.com/mechanicalsea/lighthubert
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SLayerNorm(nn.LayerNorm):
    """LayerNorm: variable 1-D size
    __base__: torch.nn.LayerNorm
    """

    def __init__(
        self, normalized_shape: int, eps: float = 1e-5, elementwise_affine: bool = True
    ) -> None:
        super(SLayerNorm, self).__init__(normalized_shape, eps, elementwise_affine)
        self.staticize()

    def staticize(self):
        self.sample_normalized_shape = self.normalized_shape[0]
        self.samples = {
            "weight": self.weight,
            "bias": self.bias,
        }

    def set_sample_config(self, sample_normalized_shape: int):
        self.sample_normalized_shape = sample_normalized_shape
        self._sample_parameters()

    def _sample_parameters(self):
        if self.elementwise_affine:
            self.samples["weight"] = self.weight[: self.sample_normalized_shape]
            self.samples["bias"] = self.bias[: self.sample_normalized_shape]
        else:
            self.samples["weight"] = None
            self.samples["bias"] = None
        return self.samples

    def calc_sampled_param_num(self):
        return self.samples["weight"].numel() + self.samples["bias"].numel()

    def get_complexity(self, sequence_length):
        return sequence_length * self.sample_normalized_shape

    @property
    def weights(self):
        return self.samples["weight"] if self.elementwise_affine else None

    @property
    def biases(self):
        return self.samples["bias"] if self.elementwise_affine else None

    @property
    def normalized_shapes(self):
        if isinstance(self.sample_normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            sample_normalized_shape = (self.sample_normalized_shape,)  # type: ignore[assignment]
        else:
            sample_normalized_shape = self.sample_normalized_shape
        return tuple(sample_normalized_shape)  # type: ignore[arg-type]

    def forward(self, input: Tensor) -> Tensor:
        self._sample_parameters()
        return F.layer_norm(
            input, self.normalized_shapes, self.weights, self.biases, self.eps
        )

    def extra_repr(self) -> str:
        return (
            f"{self.normalized_shape}, eps={self.eps}, "
            f"elementwise_affine={self.elementwise_affine}"
        )

    def clone_model(self, normalized_shape: int):
        self.set_sample_config(normalized_shape)

        m = nn.LayerNorm(normalized_shape, self.eps, self.elementwise_affine)
        if m.elementwise_affine:
            m = m.to(self.weight.device)
            m = m.to(self.weight.dtype)
            m.weight.data.copy_(self.weights)
            m.bias.data.copy_(self.biases)
        return m.eval()

    @classmethod
    def build_from(cls, m: nn.LayerNorm):
        normalized_shape = m.normalized_shape
        eps = m.eps
        elementwise_affine = m.elementwise_affine
        _m = cls(normalized_shape, eps, elementwise_affine)
        if _m.elementwise_affine:
            _m = _m.to(m.weight.device)
            _m = _m.to(m.weight.dtype)
            _m.weight.data.copy_(m.weight)
            _m.bias.data.copy_(m.bias)
        return _m


if __name__ == "__main__":
    m = SLayerNorm(3)
    print(m)
    print(m.weight.data, m.bias.data)
    m.set_sample_config(2)
    z = m.clone_model(2)
    x = SLayerNorm.build_from(z)
    print(m)
    print(m.weight.data, m.bias.data)
    print(z)
    print(z.weight.data, z.bias.data)
    print(x)
    print(x.weight.data, x.bias.data)
    inp = torch.rand((1, 2))
    print(torch.allclose(m(inp), z(inp)))
    print(torch.allclose(m(inp), x(inp)))
