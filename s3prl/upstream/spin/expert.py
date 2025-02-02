import logging
import sys

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from ..interfaces import UpstreamBase

SAMPLE_RATE = 16000
EXAMPLE_SEC = 5

logger = logging.getLogger(__name__)


class UpstreamExpert(UpstreamBase):
    def __init__(
        self,
        ckpt,
        feat_select: str = "hidden",
        **kwargs,
    ):
        super().__init__(**kwargs)

        # set directory of `spin/`
        sys.path.append("/home/leo1994122701/cslm/spin")
        from src.model import SpinModel

        ckpt_path = ckpt
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if "logger" in ckpt["hyper_parameters"]["trainer"]:
            del ckpt["hyper_parameters"]["trainer"]["logger"] 
        torch.save(ckpt, ckpt_path)

        self.model = SpinModel.load_from_checkpoint(ckpt_path, strict=False)
        if self.model.encoder_type == "HuBERT":
            self.model.encoder.model.feature_grad_mult = 0.0
        self.feat_select = feat_select

        assert self.feat_select in {
            "all_feats",
            "hidden",
            "disentangled",
            "logits",
            "probs",
            "last",
        }, self.feat_select

        if self.feat_select in {"logits", "probs"}:
            assert self.model.loss_type in {"SwavVQDisentangle"}
            self.model.loss_module.normalize_codebook()

        logger.info(f"Feature selection: {self.feat_select}")
        logger.info(f"Loss type: {self.model.loss_type}")

    def get_downsample_rates(self, key: str) -> int:
        return self.model.encoder_rate

    def forward(self, wavs):
        device = wavs[0].device
        wav_lengths = torch.LongTensor([len(wav) for wav in wavs]).to(device)
        wav_padding_mask = ~torch.lt(
            torch.arange(max(wav_lengths)).unsqueeze(0).to(device),
            wav_lengths.unsqueeze(1),
        )
        padded_wav = pad_sequence(wavs, batch_first=True)

        results = self.model(
            (padded_wav, wav_lengths, wav_padding_mask),
            feat_only=True,
        )

        outputs = {}

        if self.model.loss_type in {"SwavVQDisentangle"}:
            outputs["code"] = results["codes"]
            outputs["logits"] = results["logits"]
            outputs["probs"] = F.softmax(results["logits"], dim=-1)
        feat = results["repr_list"]
        feat_list = results["feat_list"]

        outputs["disentangled"] = feat

        if self.feat_select == "all_feats":
            outputs["hidden_states"] = [feat] + feat_list
        elif self.feat_select == "hidden":
            outputs["hidden_states"] = feat_list
        elif self.feat_select == "last":
            outputs["hidden_states"] = [feat_list[-1]]
        elif self.feat_select == "disentangled":
            outputs["hidden_states"] = [feat]
        elif self.feat_select == "logits":
            outputs["hidden_states"] = [results["logits"]]
        elif self.feat_select == "probs":
            outputs["hidden_states"] = [outputs["probs"]]

        outputs["last_hidden_state"] = outputs["hidden_states"][-1]

        return outputs
