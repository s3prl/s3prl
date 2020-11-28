# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ model.py ]
#   Synopsis     [ the linear model ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import torch
import torch.nn as nn
import torch.nn.functional as F


#########
# MODEL #
#########
class Model(nn.Module):
    def __init__(self, input_dim, output_class_num):
        super(Model, self).__init__()
        
        # init attributes
        self.linear = nn.Linear(input_dim, output_class_num)          


    def forward(self, features):
        predicted = self.linear(features)
        return F.log_softmax(predicted, dim=-1) # Use LogSoftmax since self.criterion combines nn.LogSoftmax() and nn.NLLLoss()
