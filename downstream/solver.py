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
from tqdm import tqdm, trange
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from dataloader import get_Dataloader
from mockingjay.solver import Solver, Tester
from utils.audio import mel_dim, num_freq, sample_rate, inv_spectrogram


class Downstream_Solver(Solver):
	''' Handler for complete training progress'''
	def __init__(self, config, paras, task):
		super(Downstream_Solver, self).__init__(config, paras)
		
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


	def load_data(self, dataset='train'):
		''' Load date for training / validation'''
		if dataset == 'train': 
			self.verbose('Loading source data from ' + self.config['solver']['data_path'])
			self.verbose('Loading phone data from ' + self.config['solver']['phone_path'])
		else: 
			self.verbose('Loading testing data ' + str(self.config['solver']['test_set']) + ' from ' + self.config['solver']['data_path'])
			self.verbose('Loading label data ' + str(self.config['solver']['test_set']) + ' from ' + self.config['solver']['phone_path'])
		setattr(self, 'dataloader', get_Dataloader(dataset, load='phone', use_gpu=self.paras.gpu, **self.config['solver']))


	def set_model(self, inference=False, with_head=False):
		self.mockingjay = Tester(self.config, self.paras)
		self.mockingjay.set_model(inference=True, with_head=False)

	def save_model(self, name, model_all=True):
		pass #TODO

	def load_model(self, inference=False, with_head=False):
		pass #TODO



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
		self.best_loss = 999.9

	def exec(self):
		pass #TODO
		
