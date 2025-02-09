import logging
from pathlib import Path

import yaml
import torch
import torch.nn as nn

from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


class UpstreamExpert(nn.Module):
    def __init__(self, ckpt: str, target: str = "latent", stack: int = 1):
        super().__init__()
        self.target = target
        self.stack = stack
        logger.info(f"The extraction target of the VAE: {target}")

        from cst.models.compress_ssl import CompressSSL

        self.model = CompressSSL.load_from_checkpoint(ckpt, map_location="cpu")

        wavs = [torch.randn(16000)]
        latent_len = self(wavs)["hidden_states"][-1].size(1)
        self.downsample_rate = round(16000 / latent_len)

    def get_downsample_rates(self, key: str) -> int:
        return self.downsample_rate

    def forward(self, wavs):
        wavs_len = torch.LongTensor([len(wav) for wav in wavs]).to(wavs[0].device)
        wavs = pad_sequence(wavs, batch_first=True)
        hs, hs_len, posteriors, latent_len = self.model.encode(wavs, wavs_len)
        latent = posteriors.mode()

        hidden_states = None
        if self.target == "latent":
            hidden_states = latent
        elif self.target == "reconstruct":
            dec, dec_len = self.model.decode(latent, latent_len)
            hidden_states = dec
        else:
            raise ValueError(f"Unsupported target: {self.target}")
        assert hidden_states is not None

        if self.stack > 1:
            bsz, seqlen, size = hidden_states.shape
            hidden_states = hidden_states[
                :, : seqlen // self.stack * self.stack, :
            ].reshape(bsz, seqlen // self.stack, size * self.stack)

        else:
            raise ValueError(f"Unsupported target: {self.target}")
        return {
            "hidden_states": [hidden_states],
        }
