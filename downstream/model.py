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
	def __init__(self, input_dim, output_dim, hidden_size=768, drop=0.2):
		super(LinearClassifier, self).__init__()
		# use sample's last dim as class num, means the labels must be in one-hot
		self.dense1 = nn.Linear(input_dim, hidden_size)
		self.dense2 = nn.Linear(hidden_size, hidden_size)
		self.drop1 = nn.Dropout(p=drop)
		self.drop2 = nn.Dropout(p=drop)

		self.out = nn.Linear(hidden_size, output_dim)
		self.act_fn = torch.nn.functional.relu
		self.out_fn = nn.Softmax()

		self.criterion = nn.CrossEntropyLoss()

	def forward(self, features, labels=None):
		# expected input shape
		# features from bert: (batch, layer, seq, feature)
		# features from baseline: (batch, seq, feature)
		# labels: (1, batch, seq)
		labels = labels.squeeze(0)

		if len(features.shape) == 4:
			# means is bert representation, where the second dim is layer num
			features = features.mean(dim=1)  # now simply average the representations over all layers

			# since the down-sampling (float length be truncated to int) and then up-sampling process
			# can cause a mismatch between the seq lenth of bert representation and that of input features
			# we truncate the final few frame to make two seq equal in length
			truncated_length = min(features.shape[1], labels.shape[1])
			features = features[:, :truncated_length, :]
			labels = labels[:, :truncated_length]

		hidden = self.dense1(features)
		hidden = self.drop1(hidden)
		hidden = self.act_fn(hidden)

		hidden = self.dense2(hidden)
		hidden = self.drop2(hidden)
		hidden = self.act_fn(hidden)

		logits = self.out(hidden)
		if labels is not None:
			# cause logits are in (batch, seq, class) and labels are in (batch, seq)
			# nn.CrossEntropyLoss expect to have (batch, class) and (batch,) as input
			# here we flatten logits and labels in order to apply nn.CrossEntropyLoss
			class_num = logits.size(-1)
			return self.criterion(logits.reshape(-1, class_num), labels.reshape(-1)), self.out_fn(logits)
		return self.out_fn(logits)
