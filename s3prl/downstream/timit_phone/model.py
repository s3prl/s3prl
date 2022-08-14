# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ model.py ]
#   Synopsis     [ the 1-hidden model ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBank(nn.Module):
    def __init__(self, input_dim, output_class_num, kernels, cnn_size, hidden_size, dropout, **kwargs):
        super(ConvBank, self).__init__()
        self.drop_p = dropout

        self.in_linear = nn.Linear(input_dim, hidden_size)
        latest_size = hidden_size

        # conv bank
        self.cnns = nn.ModuleList()
        assert len(kernels) > 0
        for kernel in kernels:
            self.cnns.append(nn.Conv1d(latest_size, cnn_size, kernel, padding=kernel//2))
        latest_size = cnn_size * len(kernels)

        self.out_linear = nn.Linear(latest_size, output_class_num)

    def forward(self, features):
        hidden = F.dropout(F.relu(self.in_linear(features)), p=self.drop_p)

        conv_feats = []
        hidden = hidden.transpose(1, 2).contiguous()
        for cnn in self.cnns:
            conv_feats.append(cnn(hidden))
        hidden = torch.cat(conv_feats, dim=1).transpose(1, 2).contiguous()
        hidden = F.dropout(F.relu(hidden), p=self.drop_p)

        predicted = self.out_linear(hidden)
        return predicted
