# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/mockingjay/expert.py ]
#   Synopsis     [ the mockingjay wrapper ]
#   Author       [ Andy T. Liu (https://github.com/andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


from typing import List, Tuple
from collections import OrderedDict

import yaml
import torch
from torch import Tensor

from ..interfaces import UpstreamBase
from .builder import PretrainedTransformer


class UpstreamExpert(UpstreamBase):
    """
    The Mockingjay wrapper
    """

    def __init__(self, ckpt, options_config=None, **kwargs):
        super().__init__(**kwargs)

        if options_config is not None:
            print(
                "[UpstreamExpert] - Using upstream expert config file from:",
                options_config,
            )
            with open(options_config, "r") as file:
                options = yaml.load(file, Loader=yaml.FullLoader)
        else:
            print("[UpstreamExpert] - Using the default upstream expert config")
            options = {
                "load_pretrain": "True",
                "no_grad": "False",
                "dropout": "default",
                "spec_aug": "False",
                "spec_aug_prev": "True",
                "output_hidden_states": "True",
                "permute_input": "False",
            }

        options["ckpt_file"] = ckpt
        options["select_layer"] = -1

        self.transformer = PretrainedTransformer(options, inp_dim=-1)
        assert hasattr(
            self.transformer, "extracter"
        ), "This wrapper only supports `on-the-fly` ckpt with built in feature extracters."
        self.transformer([torch.randn(16000)])

    def get_downsample_rates(self, key: str) -> int:
        return 160

    def forward(self, wavs):
        last_hidden_state, hidden_states = self.transformer(wavs)  # (batch_size, extracted_seqlen, feature_dim)
        return {
            "last_hidden_state": last_hidden_state,
            "hidden_states": hidden_states.unbind(dim=0),
        }
