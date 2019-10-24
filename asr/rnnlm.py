# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ asr/rnnlm.py ]
#   Synopsis     [ rnnlm for asr]
#   Author       [ Andy T. Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
#   Reference 1  [ https://github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import torch
import torch.nn as nn
import torch.nn.functional as F


"""
Language Model
"""
class RNN_LM(nn.Module):
    def __init__(self, emb_dim, h_dim, out_dim, layers=1, rnn='LSTM', dropout_rate=0.0):
        super().__init__()
        self.h_dim = h_dim
        self.emb = nn.Embedding(out_dim, emb_dim)
        self.drop1 = torch.nn.Dropout(dropout_rate)
        self.drop2 = torch.nn.Dropout(dropout_rate)
        self.rnn = getattr(nn, rnn.upper())(emb_dim, h_dim, num_layers=layers, dropout=dropout_rate, batch_first=True)
        self.out = nn.Linear(h_dim, out_dim)

    def forward(self, x, lens, hidden=None):
        embedded = self.drop2(self.emb(x))
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lens,batch_first=True)
        outputs, hidden = self.rnn(packed, hidden) # output: (seq_len, batch, hidden*n_dir)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs,batch_first=True)
        outputs = self.out(self.drop2(outputs))
        return hidden, outputs

