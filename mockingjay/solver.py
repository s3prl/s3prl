# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ mockingjay/solver.py ]
#   Synopsis     [ solvers for the mockingjay model: trainer / tester ]
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
import numpy as np
from tqdm import tqdm, trange
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from dataloader import get_Dataloader
from mockingjay.model import MockingjayConfig, MockingjayModel, MockingjayForMaskedAcousticModel
from mockingjay.optimization import BertAdam, WarmupLinearSchedule
from utils.audio import plot_spectrogram_to_numpy, plot_spectrogram
from utils.audio import mel_dim, num_freq


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
		if torch.cuda.is_available(): self.verbose('CUDA is available!')

		# path and directories
		self.exp_name = paras.name
		if self.exp_name is None:
			self.exp_name = '_'.join([paras.config.split('/')[-1].replace('.yaml',''),'sd'+str(paras.seed)])
		if not os.path.exists(paras.ckpdir): os.makedirs(paras.ckpdir)
		self.ckpdir = os.path.join(paras.ckpdir, self.exp_name)
		if not os.path.exists(self.ckpdir): os.makedirs(self.ckpdir)
		self.load = paras.load
		self.ckpt = os.path.join(paras.ckpdir, paras.ckpt)

		# model
		self.load_model_list = config['solver']['load_model_list']
		self.duo_feature = config['solver']['duo_feature']
		self.output_dim = num_freq if self.duo_feature else None # output dim is the same as input dim if not using duo features
		self.input_dim = mel_dim


	def verbose(self, msg, end='\n'):
		''' Verbose function for print information to stdout'''
		if self.paras.verbose:
			print('[SOLVER] - ', msg, end=end)


	def load_data(self, dataset='train', phone_loader=False):
		''' Load date for training / validation'''
		if dataset == 'train': 
			self.verbose('Loading source data from ' + self.config['solver']['data_path'])
			if self.duo_feature: self.verbose('Loading target data from ' + self.config['solver']['target_path'])
			if self.phone_loader: self.verbose('Loading phone data from ' + self.config['solver']['phone_path'])
		else: 
			self.verbose('Loading testing data ' + str(self.config['solver']['test_set']) + ' from ' + self.config['solver']['data_path'])
			if self.phone_loader: self.verbose('Loading label data ' + str(self.config['solver']['test_set']) + ' from ' + self.config['solver']['phone_path'])

		if phone_loader:
			setattr(self, 'dataloader', get_Dataloader(dataset, load='phone', use_gpu=self.paras.gpu, **self.config['solver']))
		elif self.duo_feature:
			setattr(self, 'dataloader', get_Dataloader(dataset, load='duo', use_gpu=self.paras.gpu, **self.config['solver']))
		else:
			setattr(self, 'dataloader', get_Dataloader(dataset, load='spec', use_gpu=self.paras.gpu, **self.config['solver']))


	def set_model(self, inference=False, with_head=False):
		self.verbose('Initializing Mockingjay model.')
		assert(self.input_dim is not None), 'Run load_data() to get input feature shape before initializing model.'
		
		# # Build the Mockingjay model with speech prediction head
		self.model_config = MockingjayConfig(self.config)
		self.dr = self.model_config.downsample_rate
		self.hidden_size = self.model_config.hidden_size
		
		if not inference or with_head:
			self.model = MockingjayForMaskedAcousticModel(self.model_config, self.input_dim, self.output_dim).to(self.device)
			self.mockingjay = self.model.Mockingjay

		if inference and not with_head:
			self.mockingjay = MockingjayModel(self.model_config, self.input_dim).to(self.device)
			self.mockingjay.eval()
		elif inference and with_head:
			self.model.eval()
		elif not inference:
			self.model.train()

			# Setup optimizer
			param_optimizer = list(self.model.named_parameters())

			no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
			optimizer_grouped_parameters = [
				{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
				{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
				]
			num_train_optimization_steps = self.total_steps // self.gradient_accumulation_steps

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
		else:
			raise NotImplementedError

		if self.load:
			self.load_model(inference=inference, with_head=with_head)


	def save_model(self, name, model_all=True):
		if model_all:
			all_states = {
				'SpecHead': self.model.SpecHead.state_dict(),
				'Mockingjay': self.mockingjay.state_dict(),
				"Optimizer": self.optimizer.state_dict(),
				"Global_step": self.global_step,
			}
		else:
			all_states = {
				'Mockingjay': self.mockingjay.state_dict(),
			}
		new_model_path = '{}/{}-{}.ckpt'.format(self.ckpdir, name, self.global_step)
		torch.save(all_states, new_model_path)
		self.model_kept.append(new_model_path)

		if len(self.model_kept) >= self.max_keep:
			os.remove(self.model_kept[0])
			self.model_kept.pop(0)


	def load_model(self, inference=False, with_head=False):
		self.verbose('Load model from {}'.format(self.ckpt))
		all_states = torch.load(self.ckpt, map_location='cpu')
		self.verbose('', end='')
		if 'SpecHead' in self.load_model_list:
			if not inference or with_head:
				try:
					self.model.SpecHead.load_state_dict(all_states['SpecHead'])
					self.verbose('[SpecHead] - Loaded')
				except: self.verbose('[SpecHead - X]')
		if 'Mockingjay' in self.load_model_list:
			try:
				state_dict = all_states['Mockingjay']
				# Load from a PyTorch state_dict
				old_keys = []
				new_keys = []
				for key in state_dict.keys():
					new_key = None
					if 'gamma' in key:
						new_key = key.replace('gamma', 'weight')
					if 'beta' in key:
						new_key = key.replace('beta', 'bias')
					if new_key:
						old_keys.append(key)
						new_keys.append(new_key)
				for old_key, new_key in zip(old_keys, new_keys):
					state_dict[new_key] = state_dict.pop(old_key)

				missing_keys = []
				unexpected_keys = []
				error_msgs = []
				# copy state_dict so _load_from_state_dict can modify it
				metadata = getattr(state_dict, '_metadata', None)
				state_dict = state_dict.copy()
				if metadata is not None:
					state_dict._metadata = metadata

				def load(module, prefix=''):
					local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
					module._load_from_state_dict(
						state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
					for name, child in module._modules.items():
						if child is not None:
							load(child, prefix + name + '.')

				load(self.mockingjay)
				if len(missing_keys) > 0:
					self.verbose("Weights of {} not initialized from pretrained model: {}".format(
						self.mockingjay.__class__.__name__, missing_keys))
				if len(unexpected_keys) > 0:
					self.verbose("Weights from pretrained model not used in {}: {}".format(
						self.mockingjay.__class__.__name__, unexpected_keys))
				if len(error_msgs) > 0:
					raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
									   self.mockingjay.__class__.__name__, "\n\t".join(error_msgs)))
				self.verbose('[Mockingjay] - Loaded')
			except: self.verbose('[Mockingjay - X]')

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


	def up_sample_frames(self, spec, return_first=False):
		if len(spec.shape) != 3: 
			spec = spec.unsqueeze(0)
			assert(len(spec.shape) == 3), 'Input should have acoustic feature of shape BxTxD'
		# spec shape: [batch_size, sequence_length // downsample_rate, output_dim * downsample_rate]
		spec_flatten = spec.view(spec.shape[0], spec.shape[1]*self.dr, spec.shape[2]//self.dr)
		if return_first: return spec_flatten[0]
		return spec_flatten # spec_flatten shape: [batch_size, sequence_length * downsample_rate, output_dim // downsample_rate]


	def down_sample_frames(self, spec):
		left_over = spec.shape[1] % self.dr
		if left_over != 0: spec = spec[:, :-left_over, :]
		spec_stacked = spec.view(spec.shape[0], spec.shape[1]//self.dr, spec.shape[2]*self.dr)
		return spec_stacked


	def position_encoding(self, seq_len, batch_size=None, padding_idx=None):
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

		if batch_size is not None:
			batch_sinusoid_table = np.repeat(sinusoid_table[np.newaxis,...], batch_size, axis=0)
			return batch_sinusoid_table # (batch_size, seq_len, hidden_size)
		else:
			return sinusoid_table  # (seq_len, hidden_size)


	def tile_representations(encoded_layers):
		'''Tile up the mockingjay representations to match the amount of input frames'''
		# Input - encoded_layers shape: [num_hidden_layers, batch_size, sequence_length, hidden_size]
		# Output - tiled_encoded_layers shape: [num_hidden_layers, batch_size, sequence_length * downsample_rate, hidden_size]
		
		if len(encoded_layers.shape) == 4:
			tiled_encoded_layers = []
		else:
			encoded_layers = [encoded_layers]

		for encoded_layer in encoded_layers: # for layers
			tiled_encoded_layer = np.zeros((encoded_layer.shape[0],
											encoded_layer.shape[1]*self.dr, 
											encoded_layer.shape[2]))
			for idx in range(len(tiled_encoded_layer)): # for batch
				for jdx in range(len(tiled_encoded_layer[idx])): # for each timestep
					for kdx in range(self.dr): # repeat and tile
						tiled_encoded_layer[idx][jdx+kdx] = copy.deepcopy(encoded_layer[idx][jdx])
			tiled_encoded_layers.append(tiled_encoded_layer)

		if len(tiled_encoded_layers) == 1:
			return tiled_encoded_layers[0] # return the only layer if only one layer is given at input
		else: return tiled_encoded_layers # else return all layers


class Trainer(Solver):
	''' Handler for complete training progress'''
	def __init__(self, config, paras):
		super(Trainer, self).__init__(config, paras)
		# Logger Settings
		self.logdir = os.path.join(paras.logdir, self.exp_name)
		self.log = SummaryWriter(self.logdir)

		# Training details
		self.apex = config['solver']['apex']
		self.log_step = config['solver']['log_step']
		self.save_step = config['solver']['save_step']
		self.total_steps = config['solver']['total_steps']
		self.mask_proportion = config['solver']['mask_proportion']
		self.learning_rate = float(self.config['optimizer']['learning_rate'])
		self.warmup_proportion = self.config['optimizer']['warmup_proportion']
		self.gradient_accumulation_steps = self.config['optimizer']['gradient_accumulation_steps']
		self.gradient_clipping = self.config['optimizer']['gradient_clipping']
		self.max_keep = config['solver']['max_keep']
		self.reset_train()

	def reset_train(self):
		self.model_kept = []
		self.global_step = 1
		self.best_loss = 999.9


	def process_MAM_data(self, source_spec, target_spec):
		"""Process training data for the masked acoustic model"""
		# Hack bucket
		assert(len(source_spec.shape) == 4), 'Bucketing should cause acoustic feature to have shape 1xBxTxD'
		assert(len(target_spec.shape) == 4), 'Bucketing should cause acoustic feature to have shape 1xBxTxD'
		source_spec = source_spec.squeeze(0)
		target_spec = target_spec.squeeze(0)

		# Down sample
		spec_masked = self.down_sample_frames(source_spec) # (batch_size, seq_len, mel_dim * dr)
		spec_stacked = self.down_sample_frames(target_spec) # (batch_size, seq_len, mel_dim * dr)
		assert(spec_masked.shape[1] == spec_stacked.shape[1])

		# Record length for each uttr
		spec_len = np.sum(np.sum(spec_stacked.data.numpy(), axis=-1) != 0, axis=-1)
		spec_len = [int(sl) for sl in spec_len]

		batch_size = spec_stacked.shape[0]
		seq_len = spec_stacked.shape[1]

		pos_enc = self.position_encoding(seq_len, batch_size) # (batch_size, seq_len, hidden_size)
		mask_label = np.zeros_like(spec_stacked)
		attn_mask = np.ones((batch_size, seq_len)) # (batch_size, seq_len)

		for idx in range(len(spec_stacked)):
			
			chose_proportion = int(spec_len[idx]*self.mask_proportion) # chooses % of the frame positions at random for prediction
			sub_mask_proportion = int(chose_proportion*0.8) # replace the i-th frame with (1) the [MASK] frame 80% of the time
			sub_rand_proportion = int(chose_proportion*0.1) # a random frame 10% of the time
			
			sample_index = random.sample(range(spec_len[idx]), chose_proportion + sub_rand_proportion) # sample the chosen_index and random frames
			chosen_index = sample_index[:chose_proportion]
			masked_index = chosen_index[:sub_mask_proportion]

			if sub_rand_proportion > 0:
				random_index = chosen_index[-sub_rand_proportion:]
				random_frames = sample_index[-sub_rand_proportion:]
				spec_masked[idx][random_index] = spec_masked[idx][random_frames]
			
			spec_masked[idx][masked_index] = 0 # mask frames to zero
			mask_label[idx][chosen_index] = 1 # the frames where gradients will be calculated on 

			# zero vectors for padding dimension
			pos_enc[idx][spec_len[idx]:] = 0  
			attn_mask[idx][spec_len[idx]:] = 0 

		spec_masked = spec_masked.to(device=self.device, dtype=torch.float32)
		pos_enc = torch.FloatTensor(pos_enc).to(device=self.device, dtype=torch.float32)
		mask_label = torch.ByteTensor(mask_label).to(device=self.device, dtype=torch.uint8)
		attn_mask = torch.FloatTensor(attn_mask).to(device=self.device, dtype=torch.float32)
		spec_stacked = spec_stacked.to(device=self.device, dtype=torch.float32)
		return spec_masked, pos_enc, mask_label, attn_mask, spec_stacked # (x, pos_enc, mask_label, attention_mask. y)


	def train(self):
		''' Training Unsupervised End-to-end Mockingjay Model'''
		self.verbose('Training set total ' + str(len(self.dataloader)) + ' batches.')

		pbar = tqdm(total=self.total_steps)
		while self.global_step <= self.total_steps:

			progress = tqdm(self.dataloader, desc="Iteration")

			for step, spec in enumerate(progress):

				if self.duo_feature:
					spec_masked, pos_enc, mask_label, attn_mask, spec_stacked = self.process_MAM_data(source_spec=spec[0], 
																									  target_spec=spec[1])
				else:
					spec_masked, pos_enc, mask_label, attn_mask, spec_stacked = self.process_MAM_data(source_spec=spec,
																									  target_spec=copy.deepcopy(spec))
				loss, pred_spec = self.model(spec_masked, pos_enc, mask_label, attn_mask, spec_stacked)
				
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

				if self.global_step % self.log_step == 0:
					# Log
					self.log.add_scalar('lr', self.optimizer.get_lr()[0], self.global_step)
					self.log.add_scalar('loss', loss.item(), self.global_step)
					progress.set_description("Loss %.4f" % loss.item())

				if self.global_step % self.save_step == 0 and loss.item() < self.best_loss:
					self.save_model('mockingjay')
					self.best_loss = loss.item()
					mask_spec = self.up_sample_frames(spec_masked[0], return_first=True)
					pred_spec = self.up_sample_frames(pred_spec[0], return_first=True)
					true_spec = self.up_sample_frames(spec_stacked[0], return_first=True)
					mask_spec = plot_spectrogram_to_numpy(mask_spec.data.cpu().numpy())
					pred_spec = plot_spectrogram_to_numpy(pred_spec.data.cpu().numpy())
					true_spec = plot_spectrogram_to_numpy(true_spec.data.cpu().numpy())
					self.log.add_image('mask_spec', mask_spec, self.global_step)
					self.log.add_image('pred_spec', pred_spec, self.global_step)
					self.log.add_image('true_spec', true_spec, self.global_step)

				pbar.update(1)
				if self.global_step >= self.total_steps: break
				else: self.global_step += 1
				
		pbar.close()
		self.reset_train()
		

class Tester(Solver):
	''' Handler for complete testing progress'''
	def __init__(self, config, paras):
		super(Tester, self).__init__(config, paras)
		self.dump_dir = str(self.ckpt.split('.')[0]) + '-dump/'
		if not os.path.exists(self.dump_dir): os.makedirs(self.dump_dir)


	def process_MAM_data(self, spec):
		"""Process training data for the masked acoustic model"""
		# Hack bucket
		assert(len(spec.shape) == 4), 'Bucketing should cause acoustic feature to have shape 1xBxTxD'
		spec = spec.squeeze(0)

		# Down sample
		spec_stacked = self.down_sample_frames(spec) # (batch_size, seq_len, mel_dim * dr)

		# Record length for each uttr
		spec_len = np.sum(np.sum(spec_stacked.data.numpy(), axis=-1) != 0, axis=-1)
		spec_len = [int(sl) for sl in spec_len]

		batch_size = spec_stacked.shape[0]
		seq_len = spec_stacked.shape[1]

		pos_enc = self.position_encoding(seq_len, batch_size) # (batch_size, seq_len, hidden_size)
		attn_mask = np.ones((batch_size, seq_len)) # (batch_size, seq_len)

		# zero vectors for padding dimension
		for idx in range(len(spec_stacked)):
			pos_enc[idx][spec_len[idx]:] = 0  
			attn_mask[idx][spec_len[idx]:] = 0 

		spec_stacked = spec_stacked.to(device=self.device, dtype=torch.float32)
		pos_enc = torch.FloatTensor(pos_enc).to(device=self.device, dtype=torch.float32)
		attn_mask = torch.FloatTensor(attn_mask).to(device=self.device, dtype=torch.float32)
		return spec_stacked, pos_enc, attn_mask # (x, pos_enc, attention_mask)


	def plot(self, with_head=False):
		''' Plotting the visualizations of the Unsupervised End-to-end Mockingjay Model'''
		self.verbose('Testing set total ' + str(len(self.dataloader)) + ' batches.')

		idx = 0
		for x in tqdm(self.dataloader, desc="Plotting"):
			spec_stacked, pos_enc, attn_mask = self.process_MAM_data(spec=x)
			if with_head:
				pred_spec = self.model(spec_stacked, pos_enc, attention_mask=attn_mask)

				# generate the model filled MAM spectrogram
				spec_masked = copy.deepcopy(spec_stacked)
				for i in range(len(spec_masked)):
					sample_index = random.sample(range(len(spec_masked[i])), int(len(spec_masked[i])*0.15))
					print(sample_index)
					spec_masked[i][sample_index] = 0
					print(spec_masked.shape)
				fill_spec = self.model(spec_masked, pos_enc, attention_mask=attn_mask)

				# plot reconstructed / ground-truth / MAM filled spectrogram
				for y_pred, y_true, y_fill in zip(pred_spec, spec_stacked, fill_spec):
					y_pred = self.up_sample_frames(y_pred, return_first=True)
					y_true = self.up_sample_frames(y_true, return_first=True)
					y_true = self.up_sample_frames(y_fill, return_first=True)
					plot_spectrogram(y_pred.data.cpu().numpy(), path=os.path.join(self.dump_dir, str(idx) + '_pred.png'))
					plot_spectrogram(y_true.data.cpu().numpy(), path=os.path.join(self.dump_dir, str(idx) + '_true.png'))
					plot_spectrogram(y_fill.data.cpu().numpy(), path=os.path.join(self.dump_dir, str(idx) + '_fill.png'))
					idx += 1
					if idx > 10: 
						self.verbose('Spectrogram head generated samples are saved to: {}'.format(self.dump_dir))
						exit() # visualize the first 10 testing samples
			else:
				encoded_layers = self.mockingjay(spec_stacked, pos_enc, attention_mask=attn_mask, output_all_encoded_layers=True)
				last_encoded_layer = encoded_layers[-1]

				for rep in last_encoded_layer:
					plot_spectrogram(rep.data.cpu().numpy(), path=os.path.join(self.dump_dir, str(idx) + '_hidden.png'))
					idx += 1
					if idx > 10: 
						self.verbose('Mockingjay generated samples are saved to: {}'.format(self.dump_dir))
						exit() # visualize the first 10 testing samples

	def test_phone(self):
		''' Testing Unsupervised End-to-end Mockingjay Model'''
		self.verbose('Testing set total ' + str(len(self.dataloader)) + ' batches.')

		idx = 0
		for x in tqdm(self.dataloader, desc="Testing"):
			
			spec_stacked, pos_enc, attn_mask = self.process_MAM_data(spec=x)
			encoded_layers = self.mockingjay(spec_stacked, pos_enc, attention_mask=attn_mask, output_all_encoded_layers=True)
			
			reps = tile_representations(encoded_layers)



		

