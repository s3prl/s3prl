# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ mockingjay/solver.py ]
#   Synopsis     [ solver for the mockingjay model]
#   Author       [ Andy T. Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
#   Reference 1  [ https://github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import torch
import copy
import math
import random
import itertools
import numpy as np
from tqdm import tqdm, trange
import torch.nn.functional as F
from joblib import Parallel, delayed
from tensorboardX import SummaryWriter
from dataloader import get_Dataloader
from mockingjay.model import MockingjayConfig, MockingjayForMaskedAcousticModel
from mockingjay.optimization import BertAdam, WarmupLinearSchedule


##########
# SOLVER #
##########
class Solver():
	''' Super class Solver for all kinds of tasks'''
	def __init__(self, config, paras):
		# General Settings
		self.config = config
		self.paras = paras
		self.device = torch.device('cuda') if (self.paras.gpu and torch.cuda.is_available()) else torch.device('cpu')

		self.exp_name = paras.name
		if self.exp_name is None:
			self.exp_name = '_'.join([paras.config.split('/')[-1].replace('.yaml',''),'sd'+str(paras.seed)])
		if not os.path.exists(paras.ckpdir):os.makedirs(paras.ckpdir)
		self.ckpdir = os.path.join(paras.ckpdir,self.exp_name)
		if not os.path.exists(self.ckpdir):os.makedirs(self.ckpdir)

		if torch.cuda.is_available(): self.verbose('CUDA is available!')
		self.load_model_list = config['solver']['load_model_list']
		self.max_keep = config['solver']['max_keep']
		self.reset_train()


	def verbose(self, msg, end='\r'):
		''' Verbose function for print information to stdout'''
		if self.paras.verbose:
			print('[SOLVER] - ', msg, end)

	def reset_train(self):
		self.model_kept = []
		self.global_step = 1
		self.best_loss = 999.9


	def save_model(self, name, model_all=True):
		if model_all:
			all_model = {
				'SpecHead': self.model.SpecHead.state_dict(),
				'Mockingjay': self.model.Mockingjay.state_dict(),
			}
		else:
			all_model = {
				'Mockingjay': self.model.Mockingjay.state_dict(),
			}
		new_model_path = '{}/{}-{}'.format(self.ckpdir, name, self.global_step)
		torch.save(all_model, new_model_path)
		self.model_kept.append(new_model_path)

		if len(self.model_kept) >= self.max_keep:
			os.remove(self.model_kept[0])
			self.model_kept.pop(0)


	def load_model(self, verbose=True, clf_path = None):
		verbose('Load model from {}'.format(self.ckpdir))
		all_model = torch.load(self.ckpdir)
		verbose('', end = '')
		if 'SpecHead' in self.load_model_list:
			try:
				self.model.SpecHead.load_state_dict(all_model['SpecHead'])
				verbose('[SpecHead], ', end = '')
			except: verbose('[SpecHead - X], ', end = '')
		if 'Mockingjay' in self.load_model_list:
			try:
				self.model.Mockingjay.load_state_dict(all_model['Mockingjay'])
				verbose('[Mockingjay], ', end = '')
			except: verbose('[Mockingjay - X], ', end = '')
		verbose('Loaded!')


class Trainer(Solver):
	''' Handler for complete training progress'''
	def __init__(self, config, paras):
		super(Trainer, self).__init__(config,paras)
		# Logger Settings
		self.logdir = os.path.join(paras.logdir, self.exp_name)
		self.log = SummaryWriter(self.logdir)

		# Training details
		self.apex = config['solver']['apex']
		self.save_step = config['solver']['save_step']
		self.total_epoch = config['solver']['total_epochs']
		self.mask_proportion = config['solver']['mask_proportion']
		self.learning_rate = float(self.config['optimizer']['learning_rate'])
		self.warmup_proportion = self.config['optimizer']['warmup_proportion']
		self.gradient_accumulation_steps = self.config['optimizer']['gradient_accumulation_steps']
		self.gradient_clipping = self.config['optimizer']['gradient_clipping']


	def load_data(self):
		''' Load date for training/validation'''
		self.verbose('Loading data from ' + self.config['solver']['data_path'])
		setattr(self, 'dataloader', get_Dataloader('train', load='spec', use_gpu=self.paras.gpu, **self.config['solver']))
		
		# Get 1 example for auto constructing model
		for self.x_sample in getattr(self, 'dataloader'): break
		if len(self.x_sample.shape) == 4: self.x_sample = self.x_sample[0]

	def set_model(self):
		''' Setup ASR (and CLM if enabled)'''
		self.verbose('Initializing Mockingjay model.')
		
		# # Build the Mockingjay model with speech prediction head
		self.model_config = MockingjayConfig(self.config)
		self.model = MockingjayForMaskedAcousticModel(self.model_config, self.x_sample).to(self.device)
		self.model.train()
		self.dr = self.model_config.downsample_rate
		self.hidden_size = self.model_config.hidden_size
			
		# Setup optimizer
		param_optimizer = list(self.model.named_parameters())

		no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
		optimizer_grouped_parameters = [
			{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
			{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
			]
		num_train_optimization_steps = (len(self.dataloader) // self.gradient_accumulation_steps) * self.total_epoch

		if self.apex:
			try:
				from apex.optimizers import FP16_Optimizer
				from apex.optimizers import FusedAdam
			except ImportError:
				raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

			optimizer = FusedAdam(optimizer_grouped_parameters,
								  lr=self.learning_rate,
								  bias_correction=False,
								  max_grad_norm=1.0)
			if self.config['optimizer']['loss_scale'] == 0:
				self.optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
			else:
				self.optimizer = FP16_Optimizer(optimizer, static_loss_scale=self.config['optimizer']['loss_scale'])
			self.warmup_linear = WarmupLinearSchedule(warmup=self.warmup_proportion,
													  t_total=num_train_optimization_steps)
		else:
			self.optimizer = BertAdam(optimizer_grouped_parameters,
									lr=self.learning_rate,
									warmup=self.warmup_proportion,
									t_total=num_train_optimization_steps)

		# TODO: load pre-trained model
		if self.paras.load:
			raise NotImplementedError


	def down_sample_frames(self, spec):
		left_over = spec.shape[1] % self.dr
		if left_over != 0: spec = spec[:, :-left_over, :]
		spec_stacked = spec.view(spec.shape[0], spec.shape[1]//self.dr, spec.shape[2]*self.dr)
		return spec_stacked


	def position_encoding(self, seq_len, padding_idx=None):
		''' Sinusoid position encoding table '''
		def cal_angle(position, hid_idx):
			return position / np.power(10000, 2 * (hid_idx // 2) / self.hidden_size)
	 
		def get_posi_angle_vec(position):
			return [cal_angle(position, hid_j) for hid_j in range(self.hidden_size)]
	 
		sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(seq_len)])
	 
		sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
		sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
	 
		if padding_idx is not None:
			sinusoid_table[padding_idx:] = 0. # zero vector for padding dimension
	 
		return sinusoid_table  # seq_len × hidden_size


	def process_MAM_data(self, spec):
		"""Process training data for the masked acoustic model"""
		# Hack bucket
		assert(len(spec.shape) == 4), 'Bucketing should cause acoustic feature to have shape 1xBxTxD'
		spec = spec.squeeze(0)

		# Down sample
		spec_stacked = self.down_sample_frames(spec)

		# Record length for each uttr
		spec_len = np.sum(np.sum(spec_stacked.data.numpy(), axis=-1) != 0, axis=-1)
		spec_len = [int(sl) for sl in spec_len]

		# select a proportion of frames and mask them
		spec_masked, pos_enc, mask_label, attn_mask = [], [], [], []
		for idx, frames in enumerate(spec_stacked):
			
			chose_proportion = int(spec_len[idx]*self.mask_proportion) # chooses % of the frame positions at random for prediction
			sub_mask_proportion = int(chose_proportion*0.8) # replace the i-th frame with (1) the [MASK] frame 80% of the time
			sub_rand_proportion = int(chose_proportion*0.1) # a random frame 10% of the time
			
			sample_index = random.sample(range(spec_len[idx]), chose_proportion + sub_rand_proportion) # sample the chosen_index and random frames
			random_frames = sample_index[chose_proportion:]
			chosen_index = sample_index[:chose_proportion]
			masked_index = chosen_index[:sub_mask_proportion]
			random_index = chosen_index[-sub_rand_proportion]

			x = copy.deepcopy(frames.data.numpy())
			x[random_index] = x[random_frames]
			x[masked_index] = 0
			spec_masked.append(x)

			pos_enc.append(self.position_encoding(len(x), spec_len[idx]))

			l = np.zeros((len(x), int(list(self.x_sample.shape)[-1]*self.dr))) # (seq_len, mel_dim * dr)
			l[chosen_index] = 1
			mask_label.append(l)

			a = np.ones((len(x)))
			a[spec_len[idx]:] = 0
			attn_mask.append(a)

		spec_masked = torch.FloatTensor(spec_masked).to(device=self.device, dtype=torch.float32)
		pos_enc = torch.FloatTensor(pos_enc).to(device=self.device, dtype=torch.float32)
		mask_label = torch.ByteTensor(mask_label).to(device=self.device, dtype=torch.uint8)
		attn_mask = torch.FloatTensor(attn_mask).to(device=self.device, dtype=torch.float32)
		spec_stacked = spec_stacked.to(device=self.device, dtype=torch.float32)
		return spec_masked, pos_enc, mask_label, attn_mask, spec_stacked # (x, pos_enc, mask_label, attention_mask. y)


	def exec(self):
		''' Training End-to-end ASR system'''
		self.verbose('Training set total ' + str(len(self.dataloader)) + ' batches.')

		self.reset_train()

		for epoch in trange(self.total_epoch, desc="Epoch"):
			progress = tqdm(self.dataloader, desc="Iteration")
			for step, x in enumerate(progress):

				spec_masked, pos_enc, mask_label, attn_mask, spec_stacked = self.process_MAM_data(spec=x)
				loss = self.model(spec_masked, pos_enc, mask_label, attn_mask, spec_stacked)
				
				# Accumulate Loss
				if self.gradient_accumulation_steps > 1:
					loss = loss / self.gradient_accumulation_steps
				if self.apex:
					self.optimizer.backward(loss)
				else:
					loss.backward()

				# Update
				if step % self.gradient_accumulation_steps == 0:
					if self.apex:
						# modify learning rate with special warm up BERT uses
						# if conifg.apex is False, BertAdam is used and handles this automatically
						lr_this_step = self.learning_rate * self.warmup_linear.get_lr(self.global_step, self.warmup_proportion)
						for param_group in self.optimizer.param_groups:
							param_group['lr'] = lr_this_step
					
					# Step
					grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
					if math.isnan(grad_norm):
						self.verbose('Error : grad norm is NaN @ step ' + str(self.global_step))
					else:
						self.optimizer.step()
					self.optimizer.zero_grad()

					# Log
					self.log.add_scalar('lr', self.optimizer.get_lr()[0], self.global_step)
					self.log.add_scalar('loss', loss.item(), self.global_step)
					self.global_step += 1
					progress.set_description("Loss %.4f" % loss.item())

				if self.global_step % self.save_step == 0 and loss.item() < self.best_loss:
					self.save_model('mockingjay')
					self.best_loss = loss.item()

