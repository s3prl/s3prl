import torch
import joblib
import numpy as np
import torch.nn as nn

from ..spin import hubconf


class ApplyKmeans(nn.Module):
    def __init__(self, km_path):
        super().__init__()
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np**2).sum(0, keepdims=True)

        self.register_buffer("C", torch.from_numpy(self.C_np))
        self.register_buffer("Cnorm", torch.from_numpy(self.Cnorm_np))

    def __call__(self, x):
        dist = x.pow(2).sum(1, keepdim=True) - 2 * torch.matmul(x, self.C) + self.Cnorm
        cluster_ids = dist.argmin(dim=1).cpu()
        codewords = self.C.transpose(0, 1)[cluster_ids, :]
        return codewords


class UpstreamExpert(nn.Module):
    def __init__(self, spin_name, km_path):
        super().__init__()
        self.spin = getattr(hubconf, spin_name)()
        self.km = ApplyKmeans(km_path)

    def get_downsample_rates(self, key: str) -> int:
        return self.spin.get_downsample_rates(key)

    def forward(self, wavs):
        hs = self.spin(wavs)["hidden_states"][-1]
        bs, seqlen, size = hs.shape
        hs = hs.reshape(bs * seqlen, size)
        codewords = self.km(hs)
        codewords = codewords.reshape(bs, seqlen, size)
        return {
            "hidden_states": [codewords],
        }
