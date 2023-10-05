import logging

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from .lighthubert import LightHuBERT, LightHuBERTConfig

logger = logging.getLogger(__name__)


class UpstreamExpert(torch.nn.Module):
    def __init__(self, ckpt, **kwds):
        super().__init__()

        checkpoint = torch.load(ckpt)
        assert checkpoint["cfg"]["model"]["_name"] in [
            "hubert_pruner",
            "student_hubert",
        ]
        self.cfg = LightHuBERTConfig(checkpoint["cfg"]["model"])

        if checkpoint["cfg"]["model"]["_name"] == "hubert_pruner":
            if (
                checkpoint["cfg"]["model"]["pruner_supernet"]
                .lower()
                .endswith("small.yaml")
            ):
                self.cfg.supernet_type = "small"
            elif (
                checkpoint["cfg"]["model"]["pruner_supernet"]
                .lower()
                .endswith("base.yaml")
            ):
                self.cfg.supernet_type = "base"

        self.model = LightHuBERT(self.cfg)
        self.model.load_state_dict(checkpoint["model"], strict=False)

        if checkpoint["cfg"]["model"]["_name"] == "student_hubert":
            subnet = self.model.supernet.max_subnet
        else:
            subnet = self.model.supernet.subnet
        self.model.set_sample_config(subnet)

        self.model.encoder.layerdrop = 0.0

        params = self.model.calc_sampled_param_num()
        logger.info(f"LightHubert subnet (Params {params / 1e6:.0f}M) | {subnet}")

    def get_downsample_rates(self, key: str) -> int:
        return 320

    def forward(self, wavs):
        wavs = [F.layer_norm(wav, wav.shape) for wav in wavs]

        device = wavs[0].device
        wav_lengths = torch.LongTensor([len(wav) for wav in wavs]).to(device)
        wav_padding_mask = ~torch.lt(
            torch.arange(max(wav_lengths)).unsqueeze(0).to(device),
            wav_lengths.unsqueeze(1),
        )

        hs = self.model.extract_features(
            pad_sequence(wavs, batch_first=True),
            padding_mask=wav_padding_mask,
            ret_hs=True,
        )[0]

        return {
            "hidden_states": hs,
        }
