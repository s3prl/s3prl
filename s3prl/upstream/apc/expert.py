# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/apc/expert.py ]
#   Synopsis     [ the apc wrapper ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


import torch
from torch.nn.utils.rnn import pad_packed_sequence, pad_sequence

from ..interfaces import UpstreamBase
from .apc import APC
from .audio import create_transform


class UpstreamExpert(UpstreamBase):
    def __init__(self, ckpt, **kwargs):
        super().__init__(**kwargs)

        ckpt = torch.load(ckpt, map_location="cpu")
        config = ckpt["config"]

        self.preprocessor, feat_dim = create_transform(config["data"]["audio"])
        self.model = APC(feat_dim, **config["model"]["paras"])
        self.model.load_state_dict(ckpt["model"])

        if len(self.hooks) == 0:
            self.add_hook(
                "self.model.rnn_layers[1]",
                lambda input, output: pad_packed_sequence(input[0], batch_first=True)[
                    0
                ],
            )
            self.add_hook(
                "self.model.rnn_layers[2]",
                lambda input, output: pad_packed_sequence(input[0], batch_first=True)[
                    0
                ],
            )
            self.add_hook("self.model", lambda input, output: output[1])

    def get_downsample_rates(self, key: str) -> int:
        return 160

    def forward(self, wavs):
        features = [self.preprocessor(wav.unsqueeze(0)) for wav in wavs]
        feat_lengths = [len(feat) for feat in features]

        features = pad_sequence(features, batch_first=True)
        feat_lengths = torch.LongTensor(feat_lengths)

        predicted_BxLxM, features = self.model(
            features, feat_lengths, testing=not self.training
        )

        # This forward function only does the model forward
        # The return dict is then handled by UpstreamBase's hooks
