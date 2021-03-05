import os
import math
import yaml
import random

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

import fairseq
from fairseq.models.roberta import RobertaModel

import hubconf

SAMPLE_RATE = 16000
EXAMPLE_SEC = 5


class UpstreamExpert(nn.Module):
    """
    The expert of RoBERTa
    """
    def __init__(self,
        frontend_model,
        model_name_or_path = './bert_kmeans/',
        checkpoint_file = 'bert_kmeans.pt',
        **kwargs
    ):
        super(UpstreamExpert, self).__init__()
        assert fairseq.__version__ == '0.10.2'

        self.frontend_model = frontend_model
        self.roberta = RobertaModel.from_pretrained(model_name_or_path, checkpoint_file)

        pseudo_input = torch.randn(SAMPLE_RATE * EXAMPLE_SEC)
        self.output_dim = self.forward([pseudo_input])[0].size(-1)

    # Interface
    def get_output_dim(self):
        return self.output_dim

    # Interface
    def get_downsample_rate(self):
        return 160

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

        with torch.no_grad():
            self.frontend_model.eval()
            strings = self.frontend_model(wavs)

        tokens = [self.roberta.task.source_dictionary.encode_line(string, append_eos=False, add_if_not_exist=False).long() for string in strings]
        tokens = pad_sequence(tokens, batch_first=True, padding_value=self.roberta.task.source_dictionary.pad()).to(wavs[0].device)
        features = self.roberta.extract_features(tokens)
        
        ratio = max(wav_lengths) / features.size(1)
        feat_lengths = [round(wav_len / ratio) for wav_len in wav_lengths]
        features = [feat[:length] for feat, length in zip(features, feat_lengths)]

        return features
