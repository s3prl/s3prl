"""
    Upstream expert for Distiller
    Author: Heng-Jui Chang (https://github.com/vectominist)
"""

import yaml
import torch
from ..interfaces import UpstreamBase
from .builder import PretrainedDistiller


class UpstreamExpert(UpstreamBase):
    """
    The Distiller wrapper
    """

    def __init__(self, ckpt, model_config=None, **kwargs):
        super().__init__(**kwargs)

        if model_config is not None:
            print(
                "[UpstreamExpert] - Using upstream expert config file from:",
                model_config,
            )
            with open(model_config, "r") as file:
                options = yaml.load(file, Loader=yaml.FullLoader)
        else:
            print("[UpstreamExpert] - Using the default upstream expert config")
            options = {
                "load_pretrain": "True",
                "no_grad": "False",
                "permute_input": "False",
            }

        options["ckpt_file"] = ckpt

        self.model = PretrainedDistiller(options)

    def get_downsample_rates(self, key: str) -> int:
        return 320

    def forward(self, wavs, no_pred=False):
        _, feat_final, pred, pad_mask, layer_hidden = self.model(
            wavs, get_hidden=True, no_pred=no_pred
        )
        # pred: B x N x T x D
        if not no_pred:
            hidden_feats = pred.transpose(0, 1).split(1, 0)
            hidden_feats = [hid.squeeze(0) for hid in hidden_feats]
        else:
            hidden_feats = []
        hidden_feats = [feat_final] + layer_hidden + hidden_feats

        states = {
            "last_hidden_state": None if no_pred else hidden_feats[-1],
            "hidden_states": hidden_feats,
            "pad_mask": pad_mask,
            "paper": layer_hidden[-1],  # DistilHuBERT: https://arxiv.org/abs/2110.01900
        }

        return states
