import torch
import joblib
import numpy as np
import torch.nn as nn

from ..spin import hubconf

EMBEDDING_SIZE = 256


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
        cluster_ids = dist.argmin(dim=1)
        codewords = self.C.transpose(0, 1)[cluster_ids, :]
        return codewords, cluster_ids


class UpstreamExpert(nn.Module):
    def __init__(self, spin_name, km_path, stack_num=None, cluster_num: int = 1000):
        super().__init__()
        self.spin = getattr(hubconf, spin_name)()
        self.km = ApplyKmeans(km_path)
        self.stack_num = stack_num
        self.embeddings = None
        if stack_num is not None:
            self.embeddings = nn.Embedding(cluster_num, EMBEDDING_SIZE)

    def get_downsample_rates(self, key: str) -> int:
        if self.stack_num is not None:
            return round(self.spin.get_downsample_rates(key) * self.stack_num)
        return self.spin.get_downsample_rates(key)

    def train(self, mode=True):
        if mode:
            self.eval()
            self.requires_grad_(False)
            if self.embeddings is not None:
                self.embeddings.train()
            return self
        else:
            return super().train(mode)

    def forward(self, wavs):
        hs = self.spin(wavs)["hidden_states"][-1]
        bs, seqlen, size = hs.shape
        hs = hs.reshape(bs * seqlen, size)
        codewords, cluster_ids = self.km(hs)
        codewords = codewords.reshape(bs, seqlen, size)
        cluster_ids = cluster_ids.reshape(bs, seqlen)

        if self.embeddings is None:
            return {
                "hidden_states": [codewords],
            }
        else:
            bsz, seqlen = cluster_ids.shape
            cluster_ids = (
                cluster_ids[:, : seqlen // self.stack_num * self.stack_num]
                .reshape(bsz, seqlen // self.stack_num, self.stack_num)
                .permute(2, 0, 1)
            )
            embs = []
            for idx, cluster_id in enumerate(cluster_ids):
                emb = self.embeddings(cluster_id)
                embs.append(emb)
            embs = torch.cat(embs, dim=-1)
            return {
                "hidden_states": [embs],
            }
