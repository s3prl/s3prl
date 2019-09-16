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
	def __init__(self, input_sample, out_dim, hidden_size=768):
		super(PhoneLinearClassifier, self).__init__()
		self.input_dim = input_sample.shape[-1]
		self.dense1 = nn.Linear(self.input_dim, hidden_size)
		self.dense2 = nn.Linear(hidden_size, hidden_size)
		self.dense3 = nn.Linear(hidden_size, out_dim)
		self.act_fn = torch.nn.functional.relu
		self.out_fn = nn.Softmax()
		self.criterion = nn.CrossEntropyLoss()

	def forward(self, hidden_states, labels=None):
		hidden_states = self.dense1(hidden_states)
		hidden_states = self.act_fn(hidden_states)
		hidden_states = self.dense2(hidden_states)
		hidden_states = self.act_fn(hidden_states)
		logits = self.dense3(hidden_states)
		if labels is not None:
			return self.out_fn(logits), self.criterion(logits, labels)
		return self.out_fn(logits)