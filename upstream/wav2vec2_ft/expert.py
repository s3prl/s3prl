# Copyright (c) Facebook, Inc. All Rights Reserved

# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/wav2vec2_ft/expert.py ]
#   Synopsis     [ the wav2vec2 wrapper that supports ASR-fine-tuned checkpoints ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import math
import yaml
import random
from packaging import version
#-------------#
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
#-------------#
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
    The ASR-fine-tuned wav2vec 2.0 wrapper
    """

    def __init__(self, ckpt, **kwargs):
        super(UpstreamExpert, self).__init__()
        assert version.parse(fairseq.__version__) >= version.parse("0.10.2")

        # fix CTC-finetuned head weights
        # This fix is only compatible with fairseq==0.10.1
        loaded_ckpt = torch.load(ckpt)
        proj_in_dim = loaded_ckpt['args'].w2v_args.encoder_embed_dim
        # https://github.com/pytorch/fairseq/blob/8a0b56efeecba6941c1c0e81a3b86f0a219e6653/fairseq/models/wav2vec/wav2vec2_asr.py#L362
        proj_out_dim = loaded_ckpt['args'].decoder_embed_dim
        weight_shape = (proj_out_dim, proj_in_dim)
        bias_shape = (proj_out_dim)
        loaded_ckpt['model']['w2v_encoder.proj.weight'] = torch.zeros(proj_out_dim, proj_in_dim)
        loaded_ckpt['model']['w2v_encoder.proj.bias'] = torch.zeros(proj_out_dim)
        torch.save(loaded_ckpt, ckpt)

        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt])
        self.model = model[0].w2v_encoder.w2v_model

        pseudo_input = torch.randn(1, SAMPLE_RATE * EXAMPLE_SEC)
        padding_mask = torch.zeros(1, SAMPLE_RATE * EXAMPLE_SEC).long().bool()
        pseudo_feature, padding_mask = self.model.extract_features(pseudo_input, padding_mask)

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
        
        features, feat_padding_mask = self.model.extract_features(padded_wav, wav_padding_mask)
        feat_lengths = (features.size(1) - feat_padding_mask.sum(dim=-1)).tolist()

        features = [feat[:length] for feat, length in zip(features, feat_lengths)]
        return features
