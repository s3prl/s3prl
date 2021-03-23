# Copyright (c) Facebook, Inc. All Rights Reserved

# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/hubert/expert.py ]
#   Synopsis     [ the HuBERT wrapper ]
#   Author       [ Kushal Lakhotia ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
from packaging import version

# -------------#
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

# -------------#
import fairseq


############
# CONSTANT #
############
SAMPLE_RATE = 16000
EXAMPLE_SEC = 5


###################
# UPSTREAM EXPERT #
###################
class UpstreamExpert(nn.Module):
    """
    HuBERT wrapper
    """

    def __init__(self, ckpt, **kwargs):
        super(UpstreamExpert, self).__init__()
        assert version.parse(fairseq.__version__) >= version.parse("0.10.2")

        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt])
        self.model = model[0]
        self.task = task

        pseudo_input = torch.randn(1, SAMPLE_RATE * EXAMPLE_SEC)
        pseudo_feature, padding_mask = self.model.extract_features(
            pseudo_input,
            padding_mask=None,
            mask=None,
        )

        self.output_dim = pseudo_feature.size(-1)

    # Interface
    def get_output_dim(self):
        return self.output_dim

    # Interface
    def get_downsample_rate(self):
        return 320

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
        device = wavs[0].device
        wav_lengths = torch.LongTensor([len(wav) for wav in wavs]).to(device)
        wav_padding_mask = ~torch.lt(
            torch.arange(max(wav_lengths)).unsqueeze(0).to(device),
            wav_lengths.unsqueeze(1)
        )
        padded_wav = pad_sequence(wavs, batch_first=True)

        if self.task.cfg.normalize:
            padded_wav = F.layer_norm(padded_wav, padded_wav.shape)

        features, feat_padding_mask = self.model.extract_features(
            padded_wav,
            padding_mask=wav_padding_mask,
            mask=None,
        )
        feat_lengths = (features.size(1) - feat_padding_mask.sum(dim=-1)).tolist()
        features = [feat[:length] for feat, length in zip(features, feat_lengths)]

        return features
