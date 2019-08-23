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
import numpy as np
import itertools
from tensorboardX import SummaryWriter
from joblib import Parallel, delayed
from tqdm import tqdm
import torch.nn.functional as F
from dataset import LoadDataset


VAL_STEP = 30        # Additional Inference Timesteps to run during validation (to calculate CER)
TRAIN_WER_STEP = 250 # steps for debugging info.
GRAD_CLIP = 5
CLM_MIN_SEQ_LEN = 5


class Solver():
	''' Super class Solver for all kinds of tasks'''
	def __init__(self,config,paras):
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


	def verbose(self,msg):
		''' Verbose function for print information to stdout'''
		if self.paras.verbose:
			print('[INFO]',msg)
   
	def progress(self,msg):
		''' Verbose function for updating progress on stdout'''
		if self.paras.verbose:
			print(msg+'                              ',end='\r')


class Trainer(Solver):
	''' Handler for complete training progress'''
	def __init__(self,config,paras):
		super(Trainer, self).__init__(config,paras)
		# Logger Settings
		self.logdir = os.path.join(paras.logdir,self.exp_name)
		self.log = SummaryWriter(self.logdir)
		self.valid_step = config['solver']['dev_step']
		self.best_val_ed = 2.0

		# Training details
		self.step = 0
		self.max_step = config['solver']['total_steps']
		self.tf_start = config['solver']['tf_start']
		self.tf_end = config['solver']['tf_end']
		self.apex = config['solver']['apex']

		# CLM option
		self.apply_clm = config['clm']['enable']

	def load_data(self):
		''' Load date for training/validation'''
		self.verbose('Loading data from '+self.config['solver']['data_path'])
		setattr(self,'train_set',LoadDataset('train', load='all', use_gpu=self.paras.gpu, **self.config['solver']))
		
		# Get 1 example for auto constructing model
		for self.sample_x,_ in getattr(self,'train_set'):break
		if len(self.sample_x.shape)==4: self.sample_x=self.sample_x[0]

	def set_model(self):
		''' Setup ASR (and CLM if enabled)'''
		self.verbose('[Solver] - Initializing Mockingjay model.')
		
		# # Build attention end-to-end ASR
		# self.asr_model = Seq2Seq(self.sample_x,self.mapper.get_dim(),self.config['asr_model']).to(self.device)
		# if 'VGG' in self.config['asr_model']['encoder']['enc_type']:
		# 	self.verbose('VCC Extractor in Encoder is enabled, time subsample rate = 4.')
		# self.seq_loss = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='none').to(self.device)#, reduction='none')
		
		# # Involve CTC
		# self.ctc_loss = torch.nn.CTCLoss(blank=0, reduction='mean')
		# self.ctc_weight = self.config['asr_model']['optimizer']['joint_ctc']
		
		# # TODO: load pre-trained model
		# if self.paras.load:
		# 	raise NotImplementedError
			
		# # Setup optimizer
		# if self.apex and self.config['asr_model']['optimizer']['type']=='Adam':
		# 	import apex
		# 	self.asr_opt = apex.optimizers.FusedAdam(self.asr_model.parameters(), lr=self.config['asr_model']['optimizer']['learning_rate'])
		# else:
		# 	self.asr_opt = getattr(torch.optim,self.config['asr_model']['optimizer']['type'])
		# 	self.asr_opt = self.asr_opt(self.asr_model.parameters(), lr=self.config['asr_model']['optimizer']['learning_rate'],eps=1e-8)


	def exec(self):
		''' Training End-to-end ASR system'''
		self.verbose('Training set total ' + str(len(self.train_set)) + ' batches.')

		while self.step < self.max_step:
			for x,y in self.train_set:
				self.progress('Training step - '+str(self.step))
				
				
				# Hack bucket, record state length for each uttr, get longest label seq for decode step
				assert len(x.shape) == 4,'Bucketing should cause acoustic feature to have shape 1xBxTxD'
				print(x.shape)
				x = x.squeeze(0).to(device=self.device, dtype=torch.float32)
				print(x.shape)

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

				# # CTC loss on CTC decoder
				# if self.ctc_weight>0:
				# 	target_len = torch.sum(y!=0,dim=-1)
				# 	ctc_loss = self.ctc_loss( F.log_softmax( ctc_pred.transpose(0,1),dim=-1), label, torch.LongTensor(state_len), target_len)
				# 	loss_log['train_ctc'] = ctc_loss
				
				# asr_loss = (1-self.ctc_weight)*att_loss+self.ctc_weight*ctc_loss
				# loss_log['train_full'] = asr_loss

				# # Backprop
				# asr_loss.backward()
				# grad_norm = torch.nn.utils.clip_grad_norm_(self.asr_model.parameters(), GRAD_CLIP)
				# if math.isnan(grad_norm):
				# 	self.verbose('Error : grad norm is NaN @ step '+str(self.step))
				# else:
				# 	self.asr_opt.step()
				
				# Logger
				self.write_log('loss',loss_log)
				if self.ctc_weight<1:
					self.write_log('acc',{'train':cal_acc(att_pred,label)})
				if self.step % TRAIN_WER_STEP ==0:
					self.write_log('error rate',
								   {'train':cal_cer(att_pred,label,mapper=self.mapper)})

				self.step+=1
				if self.step > self.max_step:break
	

	def write_log(self,val_name,val_dict):
		'''Write log to TensorBoard'''
		if 'att' in val_name:
			self.log.add_image(val_name,val_dict,self.step)
		elif 'txt' in val_name or 'hyp' in val_name:
			self.log.add_text(val_name, val_dict, self.step)
		else:
			self.log.add_scalars(val_name,val_dict,self.step)

