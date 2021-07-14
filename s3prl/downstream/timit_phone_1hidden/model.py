# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ model.py ]
#   Synopsis     [ the 1-hidden model ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
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
    def __init__(self, input_dim, output_class_num, hidden_size, dropout, **kwargs):
        super(Model, self).__init__()
        
        # init attributes
        self.in_linear = nn.Linear(input_dim, hidden_size)    
        self.out_linear = nn.Linear(hidden_size, output_class_num)
        self.drop = nn.Dropout(dropout)    
        self.act_fn = nn.functional.relu      


    def forward(self, features):
        hidden = self.in_linear(features)
        hidden = self.drop(hidden)
        hidden = self.act_fn(hidden)
        predicted = self.out_linear(hidden)
        return predicted
