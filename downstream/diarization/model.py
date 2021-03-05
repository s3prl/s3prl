# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ model.py ]
#   Synopsis     [ Baseline model for speaker diarization ]
#   Author       [ Jiatong ]
#   Copyright    [ Copyright(c), Johns Hopkins University ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import torch
import torch.nn as nn


#########
# MODEL #
#########
class Model(nn.Module):
    def __init__(self, input_dim, output_class_num, rnn_layers, hidden_size, **kwargs):
        super(Model, self).__init__()

        # init attributes
        self.use_rnn = rnn_layers > 0
        if self.use_rnn:
            self.rnn = nn.LSTM(
                input_dim, hidden_size, num_layers=rnn_layers, batch_first=True
            )
            self.linear = nn.Linear(hidden_size, output_class_num)
        else:
            self.linear = nn.Linear(input_dim, output_class_num)

    def forward(self, features):
        features = features.float()
        if self.use_rnn:
            hidden, _ = self.rnn(features)
            predicted = self.linear(hidden)
        else:
            predicted = self.linear(features)
        return predicted
