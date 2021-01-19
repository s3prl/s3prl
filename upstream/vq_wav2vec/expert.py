import os
import math
import yaml
import random

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

SAMPLE_RATE = 16000
EXAMPLE_SEC = 5


class UpstreamExpert(nn.Module):
    """
    The expert of vq-Wav2vec
    """

    def __init__(self, ckpt, feature_selection, **kwargs):
        super(UpstreamExpert, self).__init__()
        self.feature_selection = feature_selection

        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt])
        self.model = model[0]

        pseudo_input = torch.randn(1, SAMPLE_RATE * EXAMPLE_SEC)
        z = self.model.feature_extractor(pseudo_input)
        # z: (batch_size, feat_dim, seqlen)

        if self.feature_selection == 'codewords':
            codewords, _ = self.model.vector_quantizer.forward_idx(z)
            # codewords: (batch_size, feat_dim, seqlen) in torch.FloatTensor

        pseudo_features = eval(self.feature_selection).transpose(1, 2)
        self.output_dim = pseudo_features.size(-1)

    def _forward_embedding(self, codes):
        # codes: (batch_size, seqlen, 2)
        codes = codes.unbind(dim=-1)
        features = []
        for code, embedding in zip(codes, self.embeddings):
            features.append(embedding(code))
        return torch.stack(features, dim=0).sum(dim=0).transpose(1, 2)

    # Interface
    def get_output_dim(self):
        return self.output_dim

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
        wav_lengths = [len(wav) for wav in wavs]
        padded_wav = pad_sequence(wavs, batch_first=True)
        z = self.model.feature_extractor(padded_wav)
        # z: (batch_size, feat_dim, seqlen)

        if self.feature_selection == 'codewords':
            codewords, _ = self.model.vector_quantizer.forward_idx(z)
            # codewords: (batch_size, feat_dim, seqlen) in torch.FloatTensor

        features = eval(self.feature_selection).transpose(1, 2)
        ratio = padded_wav.size(1) / features.size(1)
        feat_lengths = [round(wav_len / ratio) for wav_len in wav_lengths]

        features = [feat[:length] for feat, length in zip(features, feat_lengths)]
        return features
