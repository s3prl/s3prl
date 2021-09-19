from collections import OrderedDict
from typing import List, Union, Dict

import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from ..interfaces import UpstreamBase

HIDDEN_DIM = 8


class UpstreamExpert(UpstreamBase):
    def __init__(self, ckpt: str = None, model_config: str = None, **kwargs):
        """
        Args:
            ckpt:
                The checkpoint path for loading your pretrained weights.
                Can be assigned by the -k option in run_downstream.py

            model_config:
                The config path for constructing your model.
                Might not needed if you also save that in your checkpoint file.
                Can be assigned by the -g option in run_downstream.py
        """
        # Pass kwargs into UpstreamBase to enable features shared across upstreams
        super().__init__(**kwargs)

        # The model needs to be a nn.Module for finetuning, not required for representation extraction
        self.model1 = nn.Linear(1, HIDDEN_DIM)
        self.model2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)

    def get_downsample_rates(self, key: str) -> int:
        """
        Since we do not do any downsampling in this example upstream
        All keys' corresponding representations have downsample rate of 1
        """
        return 1

    def forward(
        self, wavs: List[Tensor]
    ) -> Dict[str, Union[Tensor, List[Tensor]]]:
        """
        When the returning Dict contains the List with more than one Tensor,
        those Tensors should be in the same shape to train a weighted-sum on them.
        """

        wavs = pad_sequence(wavs, batch_first=True).unsqueeze(-1)
        # wavs: (batch_size, max_len, 1)

        hidden = self.model1(wavs)
        # hidden: (batch_size, max_len, hidden_dim)

        feature = self.model2(hidden)
        # feature: (batch_size, max_len, hidden_dim)

        # The "hidden_states" key will be used as default in many cases
        # Others keys in this example are presented for SUPERB Challenge
        return {
            "hidden_states": [hidden, feature],
            "PR": [hidden, feature],
            "ASR": [hidden, feature],
            "QbE": [hidden, feature],
            "SID": [hidden, feature],
            "ASV": [hidden, feature],
            "SD": [hidden, feature],
            "ER": [hidden, feature],
            "SF": [hidden, feature],
            "SE": [hidden, feature],
            "SS": [hidden, feature],
            "secret": [hidden, feature],
        }
