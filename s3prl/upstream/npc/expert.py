# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/npc/expert.py ]
#   Synopsis     [ the npc wrapper ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


import torch
from torch.nn.utils.rnn import pad_sequence

from ..interfaces import UpstreamBase
from .npc import NPC
from .audio import create_transform


class UpstreamExpert(UpstreamBase):
    def __init__(self, ckpt, **kwargs):
        super().__init__(**kwargs)

        ckpt = torch.load(ckpt, map_location="cpu")
        config = ckpt["config"]
        self.preprocessor, feat_dim = create_transform(config["data"]["audio"])
        self.model = NPC(feat_dim, **config["model"]["paras"])
        self.model.load_state_dict(ckpt["model"])

        if len(self.hooks) == 0:
            for block_id, _ in enumerate(self.model.blocks):
                self.add_hook(
                    f"self.model.blocks[{block_id}]",
                    lambda input, output: output.transpose(1, 2),
                )

            for masked_conv_id, _ in enumerate(self.model.masked_convs):
                self.add_hook(
                    f"self.model.masked_convs[{masked_conv_id}]",
                    lambda input, output: output,
                )

            self.add_hook("self.model", lambda input, output: output[1])

    def get_downsample_rates(self, key: str) -> int:
        return 160

    def forward(self, wavs):
        features = [self.preprocessor(wav.unsqueeze(0)) for wav in wavs]
        features = pad_sequence(features, batch_first=True)

        predicted_BxLxM, features = self.model(features, testing=not self.training)

        # This forward function only does the model forward
        # The return dict is then handled by UpstreamBase's hooks
