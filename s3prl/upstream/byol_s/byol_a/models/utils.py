import logging
import re
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor


class Lambda(nn.Module):
    """[NOT USED] Custom tensorflow-like Lambda function layer."""

    def __init__(self, function):
        super(Lambda, self).__init__()
        self.function = function

    def forward(self, x: Tensor) -> Tensor:
        return self.function(x)


class NetworkCommonMixIn:
    """Common mixin for network definition."""

    def load_weight(self, weight_file, device):
        """Utility to load a weight file to a device."""

        state_dict = torch.load(weight_file, map_location=device)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        # Remove unneeded prefixes from the keys of parameters.
        weights = {}
        for k in state_dict:
            m = re.search(r"(^fc\.|\.fc\.|^features\.|\.features\.)", k)
            if m is None:
                continue
            new_k = k[m.start() :]
            new_k = new_k[1:] if new_k[0] == "." else new_k
            weights[new_k] = state_dict[k]
        # Load weights and set model to eval().
        self.load_state_dict(weights)
        self.eval()
        logging.info(
            f"Using audio embbeding network pretrained weight: {Path(weight_file).name}"
        )
        return self

    def set_trainable(self, trainable=False):
        for p in self.parameters():
            if p.requires_grad:
                p.requires_grad = trainable
