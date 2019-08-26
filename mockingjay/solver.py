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
from dataset import get_Dataloader
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


	def verbose(self, msg):
		''' Verbose function for print information to stdout'''
		if self.paras.verbose:
			print('[SOLVER]', msg)
   
	def progress(self, msg):
		''' Verbose function for updating progress on stdout'''
		if self.paras.verbose:
			print(msg + '                              ', end='\r')


class Trainer(Solver):
	''' Handler for complete training progress'''
	def __init__(self, config, paras):
		super(Trainer, self).__init__(config,paras)
		# Logger Settings
		self.logdir = os.path.join(paras.logdir, self.exp_name)
		self.log = SummaryWriter(self.logdir)
		self.valid_step = config['solver']['dev_step']
		self.best_val_ed = 2.0

		# Training details
		self.apex = config['solver']['apex']
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
		spec_masked, mask_label, attn_mask = [], [], []
		for idx, frames in enumerate(spec_stacked):
			masked_indexs = torch.Tensor(random.sample(range(spec_len[idx]), int(spec_len[idx]*self.mask_proportion))).to(torch.long)
			
			x = copy.deepcopy(frames)
			x[masked_indexs] = 0
			spec_masked.append(x)

			l = torch.zeros([spec_len[idx]])
			l[masked_indexs] = 1
			mask_label.append(l)

			a = torch.ones([spec_len[idx]])
			attn_mask.append(a)

		spec_masked = torch.Tensor(spec_masked).to(device=self.device, dtype=torch.float32)
		mask_label = torch.Tensor(mask_label).to(device=self.device, dtype=torch.float32)
		attn_mask = torch.Tensor(attn_mask).to(device=self.device, dtype=torch.float32)
		spec_stacked = spec_stacked.to(device=self.device, dtype=torch.float32)
		print(attn_mask.shape)
		exit()
		return spec_masked, mask_label, attn_mask, spec_stacked # (x, mask_label, attention_mask. y)

	def exec(self):
		''' Training End-to-end ASR system'''
		self.verbose('Training set total ' + str(len(self.dataloader)) + ' batches.')

		self.global_step = 1

		for epoch in trange(self.total_epoch, desc="Epoch"):
			for x in tqdm(self.dataloader, desc="Iteration"):
				self.progress('Training step - ' + str(self.global_step))

				self.process_MAM_data(spec=x)
				
				# Accumulate Loss
				if self.gradient_accumulation_steps > 1:
					loss = loss / self.gradient_accumulation_steps
				if self.apex:
					self.optimizer.backward(loss)
				else:
					loss.backward()

				# Update
				if step % args.gradient_accumulation_steps == 0:
					if self.apex:
						# modify learning rate with special warm up BERT uses
						# if args.fp16 is False, BertAdam is used and handles this automatically
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
					self.log.add_scalar('lr', optimizer.get_lr()[0], self.global_step)
					self.log.add_scalar('loss', loss.item(), self.global_step)
					self.global_step += 1


				# ASR forwarding 
				# self.asr_opt.zero_grad()
				# ctc_pred, state_len, att_pred, _ =  self.asr_model(x, ans_len,tf_rate=tf_rate,teacher=y,state_len=state_len)

				# # Calculate loss function
				# loss_log = {}
				# label = y[:,1:ans_len+1].contiguous()
				# ctc_loss = 0
				# att_loss = 0
				
				# # CE loss on attention decoder
				# if self.ctc_weight<1:
				# 	b,t,c = att_pred.shape
				# 	att_loss = self.seq_loss(att_pred.view(b*t,c),label.view(-1))
				# 	att_loss = torch.sum(att_loss.view(b,t),dim=-1)/torch.sum(y!=0,dim=-1)\
				# 			   .to(device = self.device,dtype=torch.float32) # Sum each uttr and devide by length
				# 	att_loss = torch.mean(att_loss) # Mean by batch
				# 	loss_log['train_att'] = att_loss


				# # Backprop
				# asr_loss.backward()
				# grad_norm = torch.nn.utils.clip_grad_norm_(self.asr_model.parameters(), GRAD_CLIP)
				# if math.isnan(grad_norm):
				# 	self.verbose('Error : grad norm is NaN @ step '+str(self.step))
				# else:
				# 	self.asr_opt.step()

