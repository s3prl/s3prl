# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ downstream/model.py ]
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
		self.dense1 = nn.Linear(input_dim, hidden_size)
		self.dense2 = nn.Linear(hidden_size, hidden_size)
		self.drop1 = nn.Dropout(p=drop)
		self.drop2 = nn.Dropout(p=drop)

		self.out = nn.Linear(hidden_size, output_dim)
		self.act_fn = torch.nn.functional.relu
		self.out_fn = nn.Softmax()

		self.criterion = nn.CrossEntropyLoss(ignore_index=-100)


	def statistic(self, probabilities, labels, label_mask):
		assert(len(probabilities.shape) > 1)
		assert(probabilities.unbind(dim=-1)[0].shape == labels.shape)
		assert(labels.shape == label_mask.shape)

		probabilities = probabilities.reshape(-1, probabilities.size(-1))
		labels = labels.reshape(-1)
		label_mask = label_mask.reshape(-1)
		valid_count = label_mask.sum()
		correct_count = ((probabilities.argmax(dim=-1) == labels).type(torch.cuda.LongTensor) * label_mask).sum()
		return correct_count, valid_count


	def forward(self, features, labels=None, label_mask=None):
		# features from mockingjay: (batch_size, layer, seq_len, feature)
		# features from baseline: (batch_size, seq_len, feature)
		# labels: (batch_size, seq_len), frame by frame classification
		labels = labels.squeeze(0)

		if len(features.shape) == 4:
			# compute mean on mockingjay representations if given features from mockingjay
			features = features.mean(dim=1)  # now simply average the representations over all layers, (batch_size, seq_len, feature)

			# since the down-sampling (float length be truncated to int) and then up-sampling process
			# can cause a mismatch between the seq lenth of mockingjay representation and that of label
			# we truncate the final few timestamp of label to make two seq equal in length
			labels = labels[:, :features.size(1)]
			label_mask = label_mask[:, :features.size(1)]

		hidden = self.dense1(features)
		hidden = self.drop1(hidden)
		hidden = self.act_fn(hidden)

		hidden = self.dense2(hidden)
		hidden = self.drop2(hidden)
		hidden = self.act_fn(hidden)

		logits = self.out(hidden)
		
		if labels is not None:
			assert label_mask is not None, 'When frame-wise labels are provided, validity of each timestamp should also be provided'
			labels_with_ignore_index = 100 * (label_mask - 1) + labels * label_mask

			# cause logits are in (batch, seq, class) and labels are in (batch, seq)
			# nn.CrossEntropyLoss expect to have (N, class) and (N,) as input
			# here we flatten logits and labels in order to apply nn.CrossEntropyLoss
			class_num = logits.size(-1)
			loss = self.criterion(logits.reshape(-1, class_num), labels_with_ignore_index.reshape(-1))
			probabilities = self.out_fn(logits)

			# statistic for accuracy
			correct, valid = self.statistic(probabilities, labels, label_mask)

			return loss, probabilities.detach(), correct.detach(), valid.detach()

		return self.out_fn(logits)
