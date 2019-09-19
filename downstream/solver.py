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
from downstream.model import LinearClassifier
from utils.audio import mel_dim, num_freq, sample_rate, inv_spectrogram


##########
# SOLVER #
##########
class Downstream_Solver(Solver):
	''' Handler for complete training progress'''
	def __init__(self, config, paras, task):
		super(Downstream_Solver, self).__init__(config, paras)
		
		self.task = task

		# path and directories
		self.exp_name = self.exp_name.replace('mockingjay', task)
		paras.ckpdir = paras.ckpdir.replace('mockingjay', task)
		if not os.path.exists(paras.ckpdir): os.makedirs(paras.ckpdir)
		self.ckpdir = self.ckpdir.replace('mockingjay', task)
		if not os.path.exists(self.ckpdir): os.makedirs(self.ckpdir)
		self.load = paras.load
		self.ckpt = self.ckpt.replace('mockingjay', task)

		# model
		self.load_model_list = config['downstream']['load_model_list']
		self.run_mockingjay = True if 'mockingjay' in task else False
		if self.run_mockingjay: self.verbose('Using Mockingjay representations.')


	def load_data(self, split='train', load='phone'):
		''' Load date for training / validation'''
		assert(load in ['phone', 'sentiment', 'speaker']), 'Unsupported dataloader!'
		if load == 'phone':
			if split == 'train':
				self.verbose('Loading source data from ' + self.config['dataloader']['data_path'])
				self.verbose('Loading phone data from ' + self.config['dataloader']['phone_path'])
			else: 
				self.verbose('Loading testing data ' + str(self.config['dataloader']['test_set']) + ' from ' + self.config['dataloader']['data_path'])
				self.verbose('Loading label data ' + str(self.config['dataloader']['test_set']) + ' from ' + self.config['dataloader']['phone_path'])
		elif load == 'sentiment':
			sentiment_path = self.config['dataloader']['sentiment_path']
			self.verbose(f'Loading {split} data from {sentiment_path}')
		else:
			assert False, 'Not yet support other downstream tasks'

		setattr(self, 'dataloader', get_Dataloader(split, load=load, use_gpu=self.paras.gpu, **self.config['dataloader']))


	def set_model(self, inference=False):
		self.mockingjay = Tester(self.config, self.paras)
		self.mockingjay.set_model(inference=True, with_head=False)
		
		input_dim = self.input_dim if 'mockingjay' not in self.task else self.config['mockingjay']['hidden_size']
		self.classifier = LinearClassifier(input_dim=input_dim,
										   output_dim=self.dataloader.dataset.class_num).to(self.device)
		self.classifier.eval() if inference else self.classifier.train()
		
		if not inference:
			self.optimizer = Adam(self.classifier.parameters(), lr=self.learning_rate, betas=(0.9, 0.999))
		if self.load:
			self.load_model(inference=inference)


	def save_model(self, name, model_all=True):
		if model_all:
			all_states = {
				'Classifier': self.classifier.state_dict(),
				"Optimizer": self.optimizer.state_dict(),
				"Global_step": self.global_step,
			}
		else:
			all_states = {
				'Classifier': self.classifier.state_dict(),
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
		self.verbose('', end='')
		
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

		self.verbose('Loaded!')


	def cal_acc(self, logits, labels):
		assert(len(logits.shape) == 3)  # (batch, seq, class)
		assert(len(labels.shape) == 2)  # (batch, seq)
		accs = []
		for step in range(logits.shape[1]):
			_, ind = torch.max(logits[:, step, :], dim=1)
			acc = torch.sum((ind == labels[:, step]).type(torch.FloatTensor)) / labels.size(0)
			accs.append(acc)
		return np.mean(accs)


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
		while self.global_step <= self.total_steps:

			for features, labels in tqdm(self.dataloader, desc="Iteration"):
				# features: (1, batch, seq, feature), tensor in CPU
				# labels: (1, batch, seq), tensor in GPU
				labels = labels.to(self.device)

				if self.run_mockingjay:
					features = self.mockingjay.exec(features)
				loss, logits = self.classifier(features, labels)
				
				# Accumulate Loss
				loss.backward()

				# Update
				self.optimizer.step()
				self.optimizer.zero_grad()

				if self.global_step % self.log_step == 0:
					# Log
					acc = self.cal_acc(logits, labels)
					self.log.add_scalar('acc', acc, self.global_step)
					self.log.add_scalar('loss', loss.item(), self.global_step)
					progress.set_description("Loss %.4f" % loss.item())

				if self.global_step % self.save_step == 0:
					self.save_model(self.task)

				pbar.update(1)
				if self.global_step > self.total_steps: break
				else: self.global_step += 1
				
		pbar.close()
		self.reset_train()


##########
# TESTER #
##########
class Downstream_Tester(Downstream_Solver):
	''' Handler for complete testing progress'''
	def __init__(self, config, paras, task):
		super(Downstream_Tester, self).__init__(config, paras, task)
		self.dump_dir = str(self.ckpt.split('.')[0]) + '-' + task + '-dump/'
		if not os.path.exists(self.dump_dir): os.makedirs(self.dump_dir)
		self.duo_feature = False # Set duo feature to False since only input mel is needed during testing
		self.load = True # Tester will load pre-trained models automatically
	
	def exec(self):
		''' Testing of downstream tasks'''
		self.verbose('Testing set total ' + str(len(self.dataloader)) + ' batches.')

		test_acc = []
		for features, labels in tqdm(self.dataloader, desc="Iteration"):

			if self.run_mockingjay:
				features = self.mockingjay.exec(features)
			logits = self.classifier(features, labels)
			test_acc.append(self.cal_acc(logits, labels))
		self.verbose('Testing set accuracy: ' + str(np.mean(test_acc)))

