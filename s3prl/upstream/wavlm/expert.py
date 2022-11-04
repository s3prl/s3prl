# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/wavlm/expert.py ]
#   Synopsis     [ the WavLM wrapper ]
#   Author       [ Microsoft ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import logging

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from ..interfaces import UpstreamBase
from .WavLM import WavLM, WavLMConfig

log = logging.getLogger(__name__)

############
# CONSTANT #
############
SAMPLE_RATE = 16000
EXAMPLE_SEC = 5


###################
# UPSTREAM EXPERT #
###################
class UpstreamExpert(UpstreamBase):
    def __init__(self, ckpt, normalize=False, **kwargs):
        super().__init__(**kwargs)
        """
        normalize (bool):
            without normalization, follow the official extraction pipeline
            with normalization, extract the features right after attention layer norm
        """

        checkpoint = torch.load(ckpt)
        self.cfg = WavLMConfig(checkpoint["cfg"])
        self.model = WavLM(self.cfg)
        self.model.load_state_dict(checkpoint["model"])

        self.model.feature_grad_mult = 0.0
        self.model.encoder.layerdrop = 0.0

        if len(self.hooks) == 0:
            module_name = "self.model.encoder.layers"
            for module_id in range(len(self.model.encoder.layers)):
                layer_norm_first = self.model.encoder.layers[module_id].layer_norm_first

                if module_id == 0:
                    if layer_norm_first:
                        if normalize:
                            log.warning(
                                "Extract the layer features right before each layer's "
                                "self-attention module, but after the pre-layernorm. "
                                "This is not the official way to extract layer-wise features, "
                                "but the extracted features can have the same numerical scale "
                                "after layernorm."
                            )
                        else:
                            log.warning(
                                "Use the official layer extraction in Fairseq. "
                                "Each layer is not on the same numerical scale."
                            )

                if layer_norm_first and normalize:
                    self.add_hook(
                        f"{module_name}[{module_id}].self_attn_layer_norm",
                        lambda input, output: output.transpose(0, 1),
                    )
                else:
                    self.add_hook(
                        f"{module_name}[{module_id}]",
                        lambda input, output: input[0].transpose(0, 1),
                    )
            self.add_hook("self.model.encoder", lambda input, output: output[0])

        self._init_layerdrop = self.model.encoder.layerdrop

    @property
    def layer_drop(self):
        return self.model.encoder.layerdrop

    def set_layer_drop(self, layerdrop: float = None):
        if isinstance(layerdrop, float):
            self.model.encoder.layerdrop = layerdrop
        elif layerdrop is None:
            self.model.encoder.layerdrop = self._init_layerdrop
        else:
            raise ValueError("layerdrop can only be float or None")

    def get_downsample_rates(self, key: str) -> int:
        return 320

    def forward(self, wavs):
        if self.cfg.normalize:
            wavs = [F.layer_norm(wav, wav.shape) for wav in wavs]

        device = wavs[0].device
        wav_lengths = torch.LongTensor([len(wav) for wav in wavs]).to(device)
        wav_padding_mask = ~torch.lt(
            torch.arange(max(wav_lengths)).unsqueeze(0).to(device),
            wav_lengths.unsqueeze(1),
        )
        padded_wav = pad_sequence(wavs, batch_first=True)

        features, feat_padding_mask = self.model.extract_features(
            padded_wav,
            padding_mask=wav_padding_mask,
            mask=False,
        )

        # This forward function only does the model forward
        # The return dict is then handled by UpstreamBase's hooks
