import torch
import joblib
import torch.nn as nn

from ..spin import hubconf

EMBEDDING_SIZE = 256


class ApplyPCA(nn.Module):
    def __init__(self, pca_path):
        super().__init__()
        self.pca_model = joblib.load(pca_path)

    def __call__(self, x):
        return self.pca_model.transform(x)


class UpstreamExpert(nn.Module):
    def __init__(self, spin_name, pca_path):
        super().__init__()
        self.spin = getattr(hubconf, spin_name)()
        self.pca = ApplyPCA(pca_path)

    def get_downsample_rates(self, key: str) -> int:
        return self.spin.get_downsample_rates(key)

    def forward(self, wavs):
        hs = self.spin(wavs)["hidden_states"][-1]
        bs, seqlen, size = hs.shape
        hs = hs.reshape(bs * seqlen, size)
        hs = torch.from_numpy(self.pca(hs.detach().cpu().numpy())).to(wavs[0].device)
        hs = hs.reshape(bs, seqlen, -1)

        return {
            "hidden_states": [hs],
        }
