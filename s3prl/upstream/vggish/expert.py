# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/vggish/expert.py ]
#   Synopsis     [ the VGGish wrapper ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""

from torch.nn.utils.rnn import pad_sequence

from ..interfaces import UpstreamBase
from .audio import waveform_to_examples
from .vggish import VGGish


class UpstreamExpert(UpstreamBase):
    def __init__(self, ckpt, **kwargs):
        super().__init__(**kwargs)
        self.model = VGGish(ckpt, **kwargs)

    def get_downsample_rates(self, key: str) -> int:
        return 16000

    def forward(self, wavs):
        device = wavs[0].device

        outputs = []
        for wav in wavs:
            # each example is in - (num_examples, 1, num_frames, num_bands)
            feature = waveform_to_examples(wav.detach().cpu().numpy())
            feature = self.model(feature.to(device))
            if feature.dim() == 1:
                feature = feature.unsqueeze(0)
            outputs.append(feature)
        outputs = pad_sequence(outputs, batch_first=True)

        return {
            "last_hidden_state": outputs,
            "hidden_states": [outputs],
        }
