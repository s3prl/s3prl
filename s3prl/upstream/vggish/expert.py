# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/vggish/expert.py ]
#   Synopsis     [ the VGGish wrapper ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""

from torch.nn.utils.rnn import pad_sequence

from ..interfaces import UpstreamBase
from .vggish import VGGish
from .audio import waveform_to_examples


class UpstreamExpert(UpstreamBase):
    def __init__(self, ckpt, **kwargs):
        super().__init__(**kwargs)

        self.model = VGGish(ckpt, **kwargs)

        """
        TODO:
        add hooks
        self.model.embeddings:
        Sequential(
            (0): Linear(in_features=12288, out_features=4096, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=4096, out_features=4096, bias=True)
            (3): ReLU(inplace=True)
            (4): Linear(in_features=4096, out_features=128, bias=True)
            (5): ReLU(inplace=True)
        )
        """
        # if len(self.hooks) == 0:
        #     self.add_hook(
        #         "self.model.embeddings[1]",
        #         lambda input, output: output,
        #     )
        #     self.add_hook(
        #         "self.model.embeddings[3]",
        #         lambda input, output: output,
        #     )
        #     self.add_hook(
        #         "self.model.embeddings[5]",
        #         lambda input, output: output,
        #     )
        #     self.add_hook("self.model", lambda input, output: output.unsqueeze(1))

    def get_downsample_rates(self, key: str) -> int:
        return 16000

    def forward(self, wavs):
        device = wavs[0].device
        
        outputs = []
        for wav in wavs:
            # each example is in - (num_examples, 1, num_frames, num_bands)
            feature = waveform_to_examples(wav.cpu().numpy())
            feature = self.model(feature.to(device))
            outputs.append(feature)
        outputs = pad_sequence(outputs, batch_first=True)
        
        return {
            "last_hidden_state": outputs,
            "hidden_states": [outputs],
        }