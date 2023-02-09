# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/byol_a/expert.py ]
#   Synopsis     [ the BYOL-Audio wrapper ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


import logging

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from .byol_a import load_yaml_config, LogMelSpectrogram, RunningNorm, AudioNTT2020Task6X

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


class UpstreamExpert(nn.Module):
    """
    The BYOL-A wrapper
    """

    def __init__(
        self,
        ckpt: str,
        model_config: str,
        norm_mean: float = None,  # Has to be a float value to continue training.
        norm_std: float = None,  # The same as above.
    ):
        super().__init__()
        config = load_yaml_config(model_config)

        # Preprocessor and normalizer.
        self.to_logmelspec = LogMelSpectrogram()
        if norm_mean is None or norm_std is None:
            print('  ** CAUTION **')
            print('  This is a run for calculating statistics of the downstream task using RunningNorm and will exit in the middle of training. **')
            print('  ** CAUTION **')
            self.normalizer = RunningNorm(epoch_samples=10_000, max_update_epochs=1, axis=[0, 1, 2]) # Use single scalar mean/std values.
        else:
            print(f'*** Using normalization statistics: mean={norm_mean}, std={norm_std} ***')
            self.normalizer = lambda x: (x - norm_mean) / norm_std

        # Load pretrained weights.
        self.model = AudioNTT2020Task6X(d=config.feature_d, n_mels=config.n_mels)
        self.model.load_weight(ckpt, device='cpu')

    # Interface
    def get_output_dim(self):
        return self.output_dim

    # Interface
    def get_downsample_rates(self, key: str) -> int:
        return 160 * 2**3 # hop_size x stride=2 for 3 layers

    # Interface
    def forward(self, wavs):
        """
        Args:
            wavs:
                list of unpadded wavs [wav1, wav2, ...]
                each wav is in torch.FloatTensor with sample rate 16000
                and already put in the device assigned by command-line args

        Return:
            features:
                list of unpadded features [feat1, feat2, ...]
                each feat is in torch.FloatTensor and already
                put in the device assigned by command-line args
        """
        self.to_logmelspec.to(wavs[0].device)
        wavs = pad_sequence(wavs, batch_first=True)
        features = self.normalizer(self.to_logmelspec(wavs)).unsqueeze(1) # (B, F, T) -> (B, 1, F, T)
        layered_features = self.model.by_layers(self.model(features, layered=True)) # [(B, T, D)] x 5
        return {
            "last_hidden_state": layered_features[-1],
            "hidden_states": layered_features,
        }
