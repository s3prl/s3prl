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
from mockingjay.solver import Solver
from utils.audio import mel_dim, num_freq, sample_rate, inv_spectrogram


class Trainer(Solver):
	''' Handler for complete training progress'''
	def __init__(self, config, paras):
		super(Trainer, self).__init__(config, paras)
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

	def set_model(self, inference=False, with_head=False):
		pass #TODO

	def save_model(self, name, model_all=True):
		pass #TODO

	def load_model(self, inference=False, with_head=False):
		pass #TODO
		
	def exec(self):
		pass #TODO
