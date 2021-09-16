# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/pase/expert.py ]
#   Synopsis     [ the pase wrapper ]
#   Author       [ santi-pdp/pase ]
#   Reference    [ https://github.com/santi-pdp/pase/blob/master/pase ]
"""*********************************************************************************************"""


from torch.nn.utils.rnn import pad_sequence

from ..interfaces import UpstreamBase
from pase.models.frontend import wf_builder


class UpstreamExpert(UpstreamBase):
    def __init__(self, ckpt, model_config, **kwargs):
        super().__init__(**kwargs)

        def build_pase(ckpt, model_config):
            pase = wf_builder(model_config)
            pase.load_pretrained(ckpt, load_last=True, verbose=False)
            return pase

        self.model = build_pase(ckpt, model_config)

    def get_downsample_rates(self, key: str) -> int:
        return 160

    def forward(self, wavs):
        wavs = pad_sequence(wavs, batch_first=True)
        wavs = wavs.unsqueeze(1)

        features = self.model(wavs)  # (batch_size, feature_dim, extracted_seqlen)
        features = features.transpose(
            1, 2
        ).contiguous()  # (batch_size, extracted_seqlen, feature_dim)

        # This forward function only does the model forward
        # The return dict is then handled by UpstreamBase's hooks
