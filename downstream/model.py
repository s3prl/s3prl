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
from torch.nn.utils.rnn import pack_padded_sequence


#####################
# LINEAR CLASSIFIER #
#####################
class LinearClassifier(nn.Module):
	def __init__(self, input_dim, class_num, task, dconfig, sequencial=False):
		super(LinearClassifier, self).__init__()
		
		output_dim = class_num
		hidden_size = dconfig['hidden_size']
		drop = dconfig['drop']
		self.sequencial = sequencial
		self.select_hidden = dconfig['select_hidden']

		if self.sequencial: 
			self.rnn = nn.GRU(input_size=input_dim, hidden_size=hidden_size, num_layers=1, dropout=0.1,
							  batch_first=True, bidirectional=False)
			self.dense1 = nn.Linear(hidden_size, hidden_size)
		else:
			self.dense1 = nn.Linear(input_dim, hidden_size)

		self.dense2 = nn.Linear(hidden_size, hidden_size)
		self.drop1 = nn.Dropout(p=drop)
		self.drop2 = nn.Dropout(p=drop)

		self.out = nn.Linear(hidden_size, output_dim)

		self.act_fn = torch.nn.functional.relu
		self.out_fn = nn.LogSoftmax(dim=-1)
		self.criterion = nn.CrossEntropyLoss(ignore_index=-100)


	def statistic(self, probabilities, labels, label_mask):
		assert(len(probabilities.shape) > 1)
		assert(probabilities.unbind(dim=-1)[0].shape == labels.shape)
		assert(labels.shape == label_mask.shape)

		valid_count = label_mask.sum()
		correct_count = ((probabilities.argmax(dim=-1) == labels).type(torch.cuda.LongTensor) * label_mask).sum()
		return correct_count, valid_count


	def forward(self, features, labels=None, label_mask=None):
		# features from mockingjay: (batch_size, layer, seq_len, feature)
		# features from baseline: (batch_size, seq_len, feature)
		# labels: (batch_size, seq_len), frame by frame classification

		if len(features.shape) == 4:
			# compute mean on mockingjay representations if given features from mockingjay
			if self.select_hidden == 'last':
				features = features[:, -1, :, :]
			elif self.select_hidden == 'first':
				features = features[:, 0, :, :]
			elif self.select_hidden == 'average':
				features = features.mean(dim=1)  # now simply average the representations over all layers, (batch_size, seq_len, feature)
			else:
				raise NotImplementedError('Feature selection mode not supported!')

			# since the down-sampling (float length be truncated to int) and then up-sampling process
			# can cause a mismatch between the seq lenth of mockingjay representation and that of label
			# we truncate the final few timestamp of label to make two seq equal in length
			labels = labels[:, :features.size(1)]
			label_mask = label_mask[:, :features.size(1)]
		
		if self.sequencial:
			features, h_n = self.rnn(features)

		hidden = self.dense1(features)
		hidden = self.drop1(hidden)
		hidden = self.act_fn(hidden)

		hidden = self.dense2(hidden)
		hidden = self.drop2(hidden)
		hidden = self.act_fn(hidden)

		logits = self.out(hidden)
		prob = self.out_fn(logits)
		
		if labels is not None:
			assert(label_mask is not None), 'When frame-wise labels are provided, validity of each timestamp should also be provided'
			labels_with_ignore_index = 100 * (label_mask - 1) + labels * label_mask

			# cause logits are in (batch, seq, class) and labels are in (batch, seq)
			# nn.CrossEntropyLoss expect to have (N, class) and (N,) as input
			# here we flatten logits and labels in order to apply nn.CrossEntropyLoss
			class_num = logits.size(-1)
			loss = self.criterion(logits.reshape(-1, class_num), labels_with_ignore_index.reshape(-1))
			
			# statistic for accuracy
			correct, valid = self.statistic(prob, labels, label_mask)

			return loss, prob.detach().cpu(), correct.detach().cpu(), valid.detach().cpu()

		return prob


class RnnClassifier(nn.Module):
	def __init__(self, input_dim, class_num, task, dconfig):
		# The class_num for regression mode should be 1

		super(RnnClassifier, self).__init__()
		self.config = dconfig

		hidden_size = self.config['hidden_size']
		drop = self.config['drop']

		self.rnn = nn.GRU(input_size=input_dim, hidden_size=hidden_size, num_layers=1, dropout=drop,
						  batch_first=True, bidirectional=False)

		self.linears = []
		self.dropouts = []
		last_dim = hidden_size
		for linear_dim in self.config['linear_dims']:
			self.linears.append(nn.Linear(last_dim, linear_dim))
			self.dropouts.append(nn.Dropout(p=drop))
			last_dim = linear_dim
		self.linears = nn.ModuleList(self.linears)
		self.dropouts = nn.ModuleList(self.dropouts)

		self.act_fn = torch.nn.functional.relu
		self.out = nn.Linear(last_dim, class_num)
		
		mode = self.config['mode']
		if mode == 'classification':
			self.out_fn = nn.LogSoftmax(dim=-1)
			self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
		elif mode == 'regression':
			self.criterion = nn.MSELoss()
		else:
			raise NotImplementedError('Only classification/regression modes are supported')


	def statistic(self, probabilities, labels):
		assert(len(probabilities.shape) > 1)
		assert(probabilities.unbind(dim=-1)[0].shape == labels.shape)

		valid_count = torch.LongTensor([len(labels)])
		correct_count = ((probabilities.argmax(dim=-1) == labels).type(torch.LongTensor)).sum()
		return correct_count, valid_count


	def forward(self, features, labels=None, valid_lengths=None):
		assert(valid_lengths is not None), 'Valid_lengths is required.'
		# features from mockingjay: (batch_size, layer, seq_len, feature)
		# features from baseline: (batch_size, seq_len, feature)
		# labels: (batch_size,), one utterance to one label
		# valid_lengths: (batch_size, )

		select_hidden = self.config['select_hidden']
		if len(features.shape) == 4:
			# compute mean on mockingjay representations if given features from mockingjay
			if select_hidden == 'last':
				features = features[:, -1, :, :]
			elif select_hidden == 'first':
				features = features[:, 0, :, :]
			elif select_hidden == 'average':
				features = features.mean(dim=1)  # now simply average the representations over all layers, (batch_size, seq_len, feature)
			else:
				raise NotImplementedError('Feature selection mode not supported!')

		packed = pack_padded_sequence(features, valid_lengths, batch_first=True, enforce_sorted=True)
		_, h_n = self.rnn(packed)
		hidden = h_n[-1, :, :]
		# cause h_n directly contains info for final states
		# it will be easier to use h_n as extracted embedding
		
		activation = self.act_fn
		for linear, dropout in zip(self.linears, self.dropouts):
			hidden = linear(hidden)
			hidden = activation(hidden)
			hidden = dropout(hidden)

		logits = self.out(hidden)

		mode = self.config['mode']
		if mode == 'classification':
			result = self.out_fn(logits)
			# result: (batch_size, class_num)
		elif mode == 'regression':
			result = logits.reshape(-1)
			# result: (batch_size, )
		
		if labels is not None:
			loss = self.criterion(result, labels)

			# statistic for accuracy
			if mode == 'classification':
				correct, valid = self.statistic(result, labels)
			elif mode == 'regression':
				# correct and valid has no meaning when in regression mode
				# just to make the outside wrapper can correctly function
				correct, valid = torch.LongTensor([1]), torch.LongTensor([1])

			return loss, result.detach().cpu(), correct, valid

		return result

