from collections import OrderedDict
from typing import Dict, List, Union

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from .cobert.models.cobert_with_teacher import CobertWithTeacherConfig, CobertWithTeacherModel


class UpstreamExpert(nn.Module):
    def __init__(self, ckpt, **kwargs):
        super().__init__()

        checkpoint = torch.load(ckpt, map_location="cpu")
        cfg = CobertWithTeacherConfig(**checkpoint["cfg"]["model"])
        model = CobertWithTeacherModel.build_model(cfg)

        # code teacher is useless in this case. remove them.
        model.code_teacher_model = None
        for k in list(checkpoint["model"].keys()):
            if "code_teacher_model" in k:
                del checkpoint["model"][k]

        # also delete ema
        del checkpoint["model"]["_ema"]
        model.load_state_dict(checkpoint["model"])

        self.model = model
        self.normalize = checkpoint["cfg"]["task"]["normalize"]

    def get_downsample_rates(self, key: str) -> int:
        return 320

    def forward(self, wavs: List[Tensor]) -> Dict[str, Union[Tensor, List[Tensor]]]:
        if self.normalize:
            wavs = [F.layer_norm(wav, wav.shape) for wav in wavs]

        device = wavs[0].device
        wav_lengths = torch.LongTensor([len(wav) for wav in wavs]).to(device)
        wav_padding_mask = ~torch.lt(
            torch.arange(max(wav_lengths)).unsqueeze(0).to(device),
            wav_lengths.unsqueeze(1),
        )
        padded_wav = pad_sequence(wavs, batch_first=True)

        results = self.model.extract_features(
            padded_wav,
            padding_mask=wav_padding_mask,
            mask=None,
        )
        layer_results = [l[0].transpose(0, 1) for l in results["layer_results"]]

        return {
            "hidden_states": layer_results,
        }
