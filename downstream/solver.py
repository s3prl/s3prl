# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ downstream/solver.py ]
#   Synopsis     [ solvers for the mockingjay downstream model: trainer / tester ]
#   Author       [ Andy T. Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import torch
import copy
import math
import random
import librosa
import numpy as np
from torch.optim import Adam
from tqdm import tqdm, trange
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from dataloader import get_Dataloader
from mockingjay.solver import Solver, Tester
from downstream.model import LinearClassifier, RnnClassifier
from utils.audio import mel_dim, num_freq, sample_rate, inv_spectrogram
from runner_apc import get_apc_model


##########
# SOLVER #
##########
class Downstream_Solver(Solver):
	''' Handler for complete training progress'''
	def __init__(self, config, paras, task):
		super(Downstream_Solver, self).__init__(config, paras)
		# Downstream task the solver is solving
		self.task = task
		self.mock_paras = copy.deepcopy(paras)
		self.mock_config = copy.deepcopy(config)
		self.mock_config['timer'] = config['timer']

		# path and directories
		self.exp_name = self.exp_name.replace('mockingjay', task)
		paras.ckpdir = paras.ckpdir.replace('mockingjay', task)
		if not os.path.exists(paras.ckpdir): os.makedirs(paras.ckpdir)
		self.ckpdir = self.ckpdir.replace('mockingjay', task)
		if not os.path.exists(self.ckpdir): os.makedirs(self.ckpdir)
		self.ckpt = os.path.join(paras.ckpdir, paras.dckpt)

		# modify log directory
		paras.logdir = paras.logdir.replace('mockingjay', task)

		# model
		self.load_model_list = config['downstream']['load_model_list']
		self.run_mockingjay = True if 'mockingjay' in task else False
		self.run_apc = True if 'apc' in task else False
		assert( not (self.run_mockingjay and self.run_apc) ), 'Mockingjay and Apc can not run at the same time!'
		if self.run_mockingjay: self.verbose('Using Mockingjay representations.')


	def load_data(self, split='train', load='phone'):
		''' Load date for training / testing'''
		assert(load in ['phone', 'sentiment', 'speaker']), 'Unsupported dataloader!'
		if load == 'phone' or load == 'speaker':
			if split == 'train':
				self.verbose('Loading source data from ' + str(self.config['dataloader']['train_set']) + ' from ' + self.config['dataloader']['data_path'])
				self.verbose('Loading phone data from ' + str(self.config['dataloader']['train_set']) + ' from ' + self.config['dataloader']['phone_path'])
			elif split == 'test': 
				self.verbose('Loading testing data ' + str(self.config['dataloader']['test_set']) + ' from ' + self.config['dataloader']['data_path'])
				self.verbose('Loading label data ' + str(self.config['dataloader']['test_set']) + ' from ' + self.config['dataloader']['phone_path'])
			else:
				raise NotImplementedError('Invalid `split` argument!')

		elif load == 'sentiment':
			sentiment_path = self.config['dataloader']['sentiment_path']
			self.verbose(f'Loading {split} data from {sentiment_path}')
		else:
			raise NotImplementedError('Unsupported downstream tasks.')

		setattr(self, 'dataloader', get_Dataloader(split, load=load, use_gpu=self.paras.gpu, \
				run_mockingjay=self.run_mockingjay, **self.config['dataloader']))


	def set_model(self, inference=False):
		self.model_type = 'linear' if 'phone' in self.task else 'rnn'
		input_dim = int(self.config['downstream'][self.model_type]['input_dim']) if \
					self.config['downstream'][self.model_type]['input_dim'] != 'None' else None
		if 'mockingjay' in self.task:
			self.mockingjay = Tester(self.mock_config, self.mock_paras)
			self.mockingjay.set_model(inference=True, with_head=False)
			self.dr = self.mockingjay.dr
			if input_dim is None:
				input_dim = self.mock_config['mockingjay']['hidden_size']
		elif 'apc' in self.task:
			self.apc = get_apc_model(path=self.paras.apc_path)
			if input_dim is None: 
				input_dim = self.mock_config['mockingjay']['hidden_size'] # use identical dim size for fair comparison
		elif 'baseline' in self.task:
			if input_dim is None: 
				input_dim = mel_dim
		else:
			raise NotImplementedError('Invalid Task!')

		if self.model_type == 'linear':
			self.classifier = LinearClassifier(input_dim=input_dim,
											class_num=self.dataloader.dataset.class_num,
											task=self.task,
											dconfig=self.config['downstream']['linear'],
											sequencial=False).to(self.device)
		elif self.model_type == 'rnn':
			self.classifier = RnnClassifier(input_dim=input_dim,
											class_num=self.dataloader.dataset.class_num,
											task=self.task,
											dconfig=self.config['downstream']['rnn']).to(self.device)

		if not inference:
			self.optimizer = Adam(self.classifier.parameters(), lr=self.learning_rate, betas=(0.9, 0.999))
			self.classifier.train()
		else:
			self.classifier.eval()

		if self.load: # This will be set to True by default when Tester is running set_model()
			self.load_model(inference=inference)


	def save_model(self, name, model_all=True):
		if model_all:
			all_states = {
				'Classifier': self.classifier.state_dict(),
				'Optimizer': self.optimizer.state_dict(),
				'Global_step': self.global_step,
				'Settings': {
					'Config': self.config,
					'Paras': self.paras,
				},
			}
		else:
			all_states = {
				'Classifier': self.classifier.state_dict(),
				'Settings': {
					'Config': self.config,
					'Paras': self.paras,
				},
			}
		new_model_path = '{}/{}-{}.ckpt'.format(self.ckpdir, name, self.global_step)
		torch.save(all_states, new_model_path)
		self.model_kept.append(new_model_path)

		if len(self.model_kept) >= self.max_keep:
			os.remove(self.model_kept[0])
			self.model_kept.pop(0)


	def load_model(self, inference=False):
		self.verbose('Load model from {}'.format(self.ckpt))
		all_states = torch.load(self.ckpt, map_location='cpu')
		
		if 'Classifier' in self.load_model_list:
			try:
				self.classifier.load_state_dict(all_states['Classifier'])
				self.verbose('[Classifier] - Loaded')
			except: self.verbose('[Classifier - X]')

		if 'Optimizer' in self.load_model_list and not inference:
			try:
				self.optimizer.load_state_dict(all_states['Optimizer'])
				for state in self.optimizer.state.values():
					for k, v in state.items():
						if torch.is_tensor(v):
							state[k] = v.cuda()
				self.verbose('[Optimizer] - Loaded')
			except: self.verbose('[Optimizer - X]')

		if 'Global_step' in self.load_model_list and not inference:
			try:
				self.global_step = all_states['Global_step']
				self.verbose('[Global_step] - Loaded')
			except: self.verbose('[Global_step - X]')

		self.verbose('Model loading complete!')


###########
# TRAINER #
###########
class Downstream_Trainer(Downstream_Solver):
	''' Handler for complete training progress'''
	def __init__(self, config, paras, task):
		super(Downstream_Trainer, self).__init__(config, paras, task)

		# Logger Settings
		self.logdir = os.path.join(paras.logdir, self.exp_name)
		self.log = SummaryWriter(self.logdir)

		# Training details
		self.log_step = config['downstream']['log_step']
		self.save_step = config['downstream']['save_step']
		self.total_steps = config['downstream']['total_steps']
		self.learning_rate = float(self.config['downstream']['learning_rate'])
		self.max_keep = config['downstream']['max_keep']
		self.reset_train()


	def reset_train(self):
		self.model_kept = []
		self.global_step = 1


	def exec(self):
		''' Training of downstream tasks'''
		self.verbose('Training set total ' + str(len(self.dataloader)) + ' batches.')

		pbar = tqdm(total=self.total_steps)
		corrects = 0
		valids = 0
		losses = 0
		while self.global_step <= self.total_steps:

			for features, labels in tqdm(self.dataloader, desc="Iteration"):
				if self.global_step > self.total_steps: break
				# features: (1, batch_size, seq_len, feature)
				# dimension of labels is depends on task and dataset, but the first dimention is always trivial due to bucketing
				# eg. (1, batch_size, seq_len) or (1, batch_size)
				labels = labels.squeeze(0).to(device=self.device, dtype=torch.long)

				if self.run_mockingjay:
					# representations shape: (batch_size, layer, seq_len, feature)
					representations = self.mockingjay.forward(features, process_from_loader=True)
					features = self.up_sample_frames(features[0].squeeze(0))
				elif self.run_apc:
					# representations shape: (batch_size, layer, seq_len, feature)
					representations = self.apc.forward(features)
					features = features.squeeze(0)
				else:
					# representations shape: (batch_size, seq_len, feature)
					features = features.squeeze(0)
					representations = features.to(device=self.device, dtype=torch.float32)

				# Since zero padding technique, some timestamps of features are not valid
				# For each timestamps, we mark 1 on valid timestamps, and 0 otherwise
				# This variable can be useful for frame-wise metric, like phoneme recognition or speaker verification
				# label_mask: (batch_size, seq_len), LongTensor
				# valid_lengths: (batch_size), LongTensor
				label_mask = (features.sum(dim=-1) != 0).type(torch.LongTensor).to(device=self.device, dtype=torch.long)
				valid_lengths = label_mask.sum(dim=1)

				if self.model_type == 'linear':
					# labels: (batch_size, seq_len)
					loss, logits, correct, valid = self.classifier(representations, labels, label_mask)
				elif self.model_type == 'rnn':
					# labels: (batch_size, )
					loss, logits, correct, valid = self.classifier(representations, labels, valid_lengths)
				else:
					raise NotImplementedError

				# Accumulate Loss
				loss.backward()

				losses += loss.detach().cpu()
				corrects += correct
				valids += valid

				# Update
				self.optimizer.step()
				self.optimizer.zero_grad()

				if self.global_step % self.log_step == 0:
					# Log
					acc = corrects.item() / valids.item()
					los = losses.item() / self.global_step
					self.log.add_scalar('acc', acc, self.global_step)
					self.log.add_scalar('loss', los, self.global_step)
					pbar.set_description("Loss %.10f" % los)

					corrects = 0
					valids = 0
					losses = 0

				if self.global_step % self.save_step == 0:
					self.save_model(self.task)

				pbar.update(1)
				self.global_step += 1
				
		pbar.close()
		self.reset_train()


##########
# TESTER #
##########
class Downstream_Tester(Downstream_Solver):
	''' Handler for complete testing progress'''
	def __init__(self, config, paras, task):
		super(Downstream_Tester, self).__init__(config, paras, task)
		self.duo_feature = False # Set duo feature to False since only input mel is needed during testing
		self.load = True # Tester will load pre-trained models automatically
	
	def exec(self):
		''' Testing of downstream tasks'''
		self.verbose('Testing set total ' + str(len(self.dataloader)) + ' batches.')

		test_acc = []
		for features, labels in tqdm(self.dataloader, desc="Iteration"):
			# features: (1, batch_size, seq_len, feature)
			# dimension of labels is depends on task and dataset, but the first dimention is always trivial due to bucketing
			labels = labels.squeeze(0).to(device=self.device, dtype=torch.long)

			if self.run_mockingjay:
				# representations shape: (batch_size, layer, seq_len, feature)
				representations = self.mockingjay.forward(features, process_from_loader=True)
				features = self.up_sample_frames(features[0].squeeze(0))
			elif self.run_apc:
				# representations shape: (batch_size, layer, seq_len, feature)
				representations = self.apc.forward(features)
				features = features.squeeze(0)
			else:
				# representations shape: (batch_size, seq_len, feature)
				features = features.squeeze(0)
				representations = features.to(device=self.device, dtype=torch.float32)

			# Since zero padding technique, some timestamps of features are not valid
			# For each timestamps, we mark 1 on valid timestamps, and 0 otherwise
			# This variable can be useful for frame-wise metric, like phoneme recognition or speaker verification
			# label_mask: (batch_size, seq_len), LongTensor
			label_mask = (features.sum(dim=-1) != 0).type(torch.LongTensor).to(device=self.device, dtype=torch.long)
			valid_lengths = label_mask.sum(dim=1)

			if self.model_type == 'linear':
				# labels: (batch_size, seq_len)
				loss, logits, correct, valid = self.classifier(representations, labels, label_mask)
			elif self.model_type == 'rnn':
				# labels: (batch_size, )
				loss, logits, correct, valid = self.classifier(representations, labels, valid_lengths)
			else:
				raise NotImplementedError
			
			test_acc.append(correct.item() / valid.item())

		test_acc = torch.FloatTensor(test_acc)
		self.verbose('Testing set accuracy: ' + str(test_acc.mean().item()))

