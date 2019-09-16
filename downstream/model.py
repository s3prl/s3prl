# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ mockingjay/downstream_model.py ]
#   Synopsis     [ Implementation of downstream models ]
#   Author       [ Andy T. Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import torch
from torch import nn


#####################
# LINEAR CLASSIFIER #
#####################
class LinearClassifier(nn.Module):
	def __init__(self, input_dim, output_sample, hidden_size=768, drop=0.2):
		super(PhoneLinearClassifier, self).__init__()
		self.output_dim = output_sample.shape[-1]
		
		self.dense1 = nn.Linear(input_dim, hidden_size)
		self.dense2 = nn.Linear(hidden_size, hidden_size)
		self.drop1 = nn.Dropout(p=drop)
		self.drop2 = nn.Dropout(p=drop)

		self.out = nn.Linear(hidden_size, out_dim)
		self.act_fn = torch.nn.functional.relu
		self.out_fn = nn.Softmax()

		self.criterion = nn.CrossEntropyLoss()

	def forward(self, hidden_states, labels=None):
		hidden_states = self.dense1(hidden_states)
		hidden_states = self.drop1(hidden_states)
		hidden_states = self.act_fn(hidden_states)

		hidden_states = self.dense2(hidden_states)
		hidden_states = self.drop2(hidden_states)
		hidden_states = self.act_fn(hidden_states)

		logits = self.out(hidden_states)
		if labels is not None:
			return self.out_fn(logits), self.criterion(logits, labels)
		return self.out_fn(logits)