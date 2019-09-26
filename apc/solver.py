# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ apc/solver.py ]
#   Synopsis     [ training and testing of the apc model ]
#   Author       [ Andy T. Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
#   Reference    [ Modified and rewrite based on: https://github.com/iamyuanchung/Autoregressive-Predictive-Coding ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import torch
import logging
import argparse
from collections import namedtuple

import numpy as np
from torch.utils import data
from torch import nn, optim
from torch.autograd import Variable
import tensorboard_logger
from tensorboard_logger import log_value

from dataloader import get_Dataloader
from apc.model import APCModel


PrenetConfig = namedtuple(
	'PrenetConfig', ['input_size', 'hidden_size', 'num_layers', 'dropout'])
RNNConfig = namedtuple(
	'RNNConfig', ['input_size', 'hidden_size', 'num_layers', 'dropout', 'residual'])


class Solver():

	def __init__(self, config):
		
		self.congig = config
		self.model_dir = os.path.join(self.config.result_path, self.config.experiment_name)
		self.log_dir = os.path.join(self.config.log_path, self.config.experiment_name)
		os.makedirs(self.model_dir, exist_ok=True)
		os.makedirs(self.log_dir, exist_ok=True)

		logging.basicConfig(
			level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
			filename=self.log_dir,
			filemode='w')

		# define a new Handler to log to console as well
		console = logging.StreamHandler()
		console.setLevel(logging.INFO)
		formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
		console.setFormatter(formatter)
		logging.getLogger('').addHandler(console)

		logging.info('Model Parameters: ')
		logging.info('Prenet Depth: %d' % (self.config.prenet_num_layers))
		logging.info('Prenet Dropout: %f' % (self.config.prenet_dropout))
		logging.info('RNN Depth: %d ' % (self.config.rnn_num_layers))
		logging.info('RNN Hidden Dim: %d' % (self.config.rnn_hidden_size))
		logging.info('RNN Residual Connections: %s' % (self.config.rnn_residual))
		logging.info('RNN Dropout: %f' % (self.config.rnn_dropout))
		logging.info('Optimizer: %s ' % (self.config.optimizer))
		logging.info('Batch Size: %d ' % (self.config.batch_size))
		logging.info('Initial Learning Rate: %f ' % (self.config.learning_rate))
		logging.info('Time Shift: %d' % (self.config.time_shift))
		logging.info('Gradient Clip Threshold: %f' % (self.config.clip_thresh))


	def verbose(self, msg, end='\n'):
		''' Verbose function for print information to stdout'''
		if self.paras.verbose:
			print('[SOLVER] - ', msg, end=end)


	def load_data(self, split='train'):
		''' Load data for training / testing'''
		if split == 'train': 
			self.verbose('Loading source data from ' + self.config.data_path)
		else: 
			self.verbose('Loading testing data ' + str(self.config.test_set) + ' from ' + self.config.data_path)
		setattr(self, 'dataloader', get_Dataloader(split, load='spec', data_path=self.config.data_path, 
												   batch_size=self.config.batch_size, 
												   max_timestep=3000, max_label_len=400, 
				   								   use_gpu=True, n_jobs=16, 
				   								   train_set=self.config.train_set, 
				   								   dev_set=self.config.dev_set, 
				   								   test_set=self.config.test_set, 
				   								   dev_batch_size=1))


	def set_model(self, inference=False):
		if self.config.prenet_num_layers == 0:
			prenet_config = None
			rnn_config = RNNConfig(
				self.config.feature_dim, self.config.rnn_hidden_size, self.config.rnn_num_layers,
				self.config.rnn_dropout, self.config.rnn_residual)
		else:
			prenet_config = PrenetConfig(
				self.config.feature_dim, self.config.rnn_hidden_size, self.config.prenet_num_layers,
				self.config.prenet_dropout)
			rnn_config = RNNConfig(
				self.config.rnn_hidden_size, self.config.rnn_hidden_size, self.config.rnn_num_layers,
				self.config.rnn_dropout, self.config.rnn_residual)

		self.model = APCModel(mel_dim=self.config.feature_dim,
							  prenet_config=prenet_config,
							  rnn_config=rnn_config).cuda()

		if inference:
			self.criterion = nn.L1Loss()
			if self.config.optimizer == 'adam':
				self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
			elif self.config.optimizer == 'adadelta':
				self.optimizer = optim.Adadelta(self.model.parameterlearning_rates())
			elif self.config.optimizer == 'sgd':
				self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.learning_rate)
			elif self.config.optimizer == 'adagrad':
				self.optimizer = optim.Adagrad(self.model.parameters(), lr=self.config.learning_rate)
			elif self.config.optimizer == 'rmsprop':
				self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.config.learning_rate)
			else:
				raise NotImplementedError("Learning method not supported for the task")

		# setup tensorboard logger
		tensorboard_logger.configure(os.path.join(self.model_dir, self.config.experiment_name + '.tb_log'))


	def load_model(self, path):
		self.verbose('Load model from {}'.format(path))
		state = torch.load(self.ckpt, map_location='cpu')
		self.verbose('', end='')
		try:
			self.model.load_state_dict(state)
			self.verbose('[SpecHead] - Loaded')
		except: self.verbose('[SpecHead - X]')


	def process_data(self, batch_x):
		assert(len(batch_x.shape) == 4), 'Bucketing should cause acoustic feature to have shape 1xBxTxD'
		with torch.no_grad()
			# Hack bucket
			batch_x = batch_x.squeeze(0)
			# compute length for each uttr
			batch_l = np.sum(np.sum(batch_x.data.numpy(), axis=-1) != 0, axis=-1)
			batch_l = [int(sl) for sl in batch_l]
		return batch_x, batch_l


	####################
	##### Training #####
	####################
	def train(self):
		
		model_kept = []
		global_step = 0

		for epoch_i in range(self.config.epochs):

			self.model.train()
			train_losses = []
			for batch_x in self.dataloader:

				batch_x, batch_l = self.process_data(batch_x)
				_, indices = torch.sort(batch_l, descending=True)

				batch_x = Variable(batch_x[indices]).cuda()
				batch_l = Variable(batch_l[indices]).cuda()

				outputs, _ = self.model(batch_x[:, :-self.config.time_shift, :], \
										batch_l - self.config.time_shift)

				self.optimizer.zero_grad()
				loss = self.criterion(outputs, batch_x[:, self.config.time_shift:, :])
				train_losses.append(loss.item())
				loss.backward()
				grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_thresh)
				self.optimizer.step()

				log_value("training loss (step-wise)", float(loss.item()), global_step)
				log_value("gradient norm", grad_norm, global_step)

				global_step += 1

			# log and save
			logging.info('Epoch: %d Training Loss: %.5f Validation Loss: %.5f' % (epoch_i + 1, np.mean(train_losses), np.mean(val_losses)))
			log_value("training loss (epoch-wise)", np.mean(train_losses), epoch_i)
			
			new_model_path = os.path.join(self.model_dir, 'apc-epoch-%d' % (epoch_i + 1) + '.ckpt')
			torch.save(self.model.state_dict(), new_model_path)
			model_kept.append(new_model_path)

			if len(model_kept) >= self.config.max_keep:
				os.remove(model_kept[0])
				model_kept.pop(0)


	###################
	##### Testing #####
	###################
	def test(self):
		self.model.eval()
		test_losses = []
		with torch.no_grad():
			for test_batch_x in self.dataloader:

				test_batch_x, test_batch_l = self.process_data(test_batch_x)
				_, test_indices = torch.sort(test_batch_l, descending=True)

				test_batch_x = Variable(test_batch_x[test_indices]).cuda()
				test_batch_l = Variable(test_batch_l[test_indices]).cuda()

				test_outputs, _ = self.model(test_batch_x[:, :-self.config.time_shift, :], \
											 test_batch_l - self.config.time_shift)

				test_loss = self.criterion(test_outputs, test_batch_x[:, self.config.time_shift:, :])
				test_losses.append(test_loss.item())

		log_value("testing loss (epoch-wise)", np.mean(test_losses), epoch_i)


	###################
	##### Forward #####
	###################
	def forward(self, batch_x, all_layers=True):
		self.model.eval()
		with torch.no_grad():

			batch_x, batch_l = self.process_data(batch_x)
			_, indices = torch.sort(batch_l, descending=True)

			batch_x = Variable(batch_x[indices]).cuda()
			batch_l = Variable(batch_l[indices]).cuda()

			_, feats = self.model(batch_x[:, :-self.config.time_shift, :], \
									batch_l - self.config.time_shift)
		# feats shape: (num_layers, batch_size, seq_len, rnn_hidden_size)
		if not all_layers:
			return feats[-1, :, :, :] # (batch_size, seq_len, rnn_hidden_size)
		else:
			return feats.permute(1, 0, 2, 3).contiguous() # (batch_size, num_layers, seq_len, rnn_hidden_size)

