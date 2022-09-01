# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ model.py ]
#   Synopsis     [ the RNN model for speech separation ]
#   Source       [ The code is from https://github.com/funcwj/uPIT-for-speech-separation ]
"""*********************************************************************************************"""

import torch
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence

class SepRNN(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 num_bins,
                 rnn="lstm",
                 num_spks=2,
                 num_layers=3,
                 hidden_size=896,
                 dropout=0.0,
                 non_linear="relu",
                 bidirectional=True):
        super(SepRNN, self).__init__()
        if non_linear not in ["relu", "sigmoid", "tanh"]:
            raise ValueError(
                "Unsupported non-linear type:{}".format(non_linear))
        self.num_spks = num_spks
        rnn = rnn.upper()
        if rnn not in ['RNN', 'LSTM', 'GRU']:
            raise ValueError("Unsupported rnn type: {}".format(rnn))
        self.rnn = getattr(torch.nn, rnn)(
            input_dim,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional)
        self.drops = torch.nn.Dropout(p=dropout)
        self.linear = torch.nn.ModuleList([
            torch.nn.Linear(hidden_size * 2
                         if bidirectional else hidden_size, num_bins)
            for _ in range(self.num_spks)
        ])
        self.non_linear = {
            "relu": torch.nn.functional.relu,
            "sigmoid": torch.nn.functional.sigmoid,
            "tanh": torch.nn.functional.tanh
        }[non_linear]
        self.num_bins = num_bins

    def forward(self, x, train=True):
        assert isinstance(x, PackedSequence)
        x, _ = self.rnn(x)
        # using unpacked sequence
        # x [bs, seq_len, feat_dim]
        x, len_x = pad_packed_sequence(x, batch_first=True)
        x = self.drops(x)
        m = []
        for linear in self.linear:
            y = linear(x)
            y = self.non_linear(y)
            if not train:
                y = y.view(-1, self.num_bins)
            m.append(y)
        return m
