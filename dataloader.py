# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ dataset.py ]
#   Synopsis     [ Librispeech dataset for solver ]
#   Author       [ Andy T. Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
#   Reference    [ https://github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import torch
import pickle
import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from utils.asr import zero_padding,target_padding

# TODO : Move this to config
HALF_BATCHSIZE_TIME = 800
HALF_BATCHSIZE_LABEL = 150


################
# LIBRIDATASET #
################
# Librispeech Dataset (work in bucketing style)
# Parameters
#     - file_path    : str, file path to dataset
#     - split        : str, data split (train / dev / test)
#     - max_timestep : int, max len for input (set to 0 for no restriction)
#     - max_label_len: int, max len for output (set to 0 for no restriction)
#     - bucket_size  : int, batch size for each bucket
#     - load         : str, types of data to load: ['all', 'text', 'spec', 'duo']
class LibriDataset(Dataset):
	def __init__(self, file_path, sets, bucket_size, max_timestep=0, max_label_len=0, drop=False, load='all'):
		# Read file
		self.root = file_path
		tables = [pd.read_csv(os.path.join(file_path, s + '.csv')) for s in sets]
		self.table = pd.concat(tables, ignore_index=True).sort_values(by=['length'], ascending=False)
		self.load = load

		# Crop seqs that are too long
		if drop and max_timestep > 0 and self.load != 'text':
			self.table = self.table[self.table.length < max_timestep]
		if drop and max_label_len > 0:
			self.table = self.table[self.table.label.str.count('_')+1 < max_label_len]


	def __getitem__(self, index):
		# Load label
		if self.load != 'spec':
			y_batch = [y for y in self.Y[index]]
			y_pad_batch = target_padding(y_batch, max([len(v) for v in y_batch]))
			if self.load == 'text':
				return y_pad_batch
		
		# Load acoustic feature and pad
		x_batch = [torch.FloatTensor(np.load(os.path.join(self.root, x_file))) for x_file in self.X[index]]
		x_pad_batch = pad_sequence(x_batch, batch_first=True)

		# Return (x_spec) 
		if self.load == 'spec':
			return x_pad_batch
		# Return (x_spec, t_spec)
		elif self.load == 'duo':
			t_batch = [torch.FloatTensor(np.load(os.path.join(self.t_root, t_file))) for t_file in self.T[index]]
			t_pad_batch = pad_sequence(t_batch, batch_first=True)
			return x_pad_batch, t_pad_batch
		elif self.load == 'phone':
			p_pad_batch = None # TODO
			return x_pad_batch, p_pad_batch
		# return (x, y)
		else:
			return x_pad_batch, y_pad_batch
			
	
	def __len__(self):
		return len(self.Y)


###############
# MEL DATASET #
###############
class MelDataset(LibriDataset):
	
	def __init__(self, file_path, sets, bucket_size, max_timestep=0, max_label_len=0, drop=False, load='all'):
		super(AsrDataset, self).__init__(file_path, sets, bucket_size, max_timestep, max_label_len, drop, load)

		assert(self.load in ['all', 'text', 'spec']), 'This dataset loads mel features and text labels.'
		X = self.table['file_path'].tolist()
		X_lens = self.table['length'].tolist()
			
		Y = [list(map(int, label.split('_'))) for label in self.table['label'].tolist()]
		if self.load == 'text':
			Y.sort(key=len,reverse=True)

		# Bucketing, X & X_len is dummy when load == 'text'
		self.X = []
		self.Y = []
		batch_x, batch_len, batch_y = [], [], []

		for x, x_len, y in zip(X, X_lens, Y):
			batch_x.append(x)
			batch_len.append(x_len)
			batch_y.append(y)
			
			# Fill in batch_x until batch is full
			if len(batch_x) == bucket_size:
				# Half the batch size if seq too long
				if (bucket_size >= 2) and ((max(batch_len) > HALF_BATCHSIZE_TIME) or (max([len(y) for y in batch_y]) > HALF_BATCHSIZE_LABEL)):
					self.X.append(batch_x[:bucket_size//2])
					self.X.append(batch_x[bucket_size//2:])
					self.Y.append(batch_y[:bucket_size//2])
					self.Y.append(batch_y[bucket_size//2:])
				else:
					self.X.append(batch_x)
					self.Y.append(batch_y)
				batch_x, batch_len, batch_y = [], [], []
		
		# Gather the last batch
		if len(batch_x) > 0:
			self.X.append(batch_x)
			self.Y.append(batch_y)


######################
# MEL LINEAR DATASET #
######################
class Mel_Linear_Dataset(LibriDataset):
	
	def __init__(self, file_path, target_path, sets, bucket_size, max_timestep=0, max_label_len=0, drop=False, load='all'):
		super(Mel_Linear_Dataset, self).__init__(file_path, sets, bucket_size, max_timestep, max_label_len, drop, load)

		assert(self.load == 'duo'), 'This dataset loads duo features: mel spectrogram and linear spectrogram.'
		# Read Target file
		self.t_root = target_path
		t_tables = [pd.read_csv(os.path.join(target_path, s + '.csv')) for s in sets]
		self.t_table = pd.concat(t_tables,ignore_index=True).sort_values(by=['length'],ascending=False)

		T = self.t_table['file_path'].tolist()
		X = self.table['file_path'].tolist()
		X_lens = self.table['length'].tolist()
			
		Y = [list(map(int, label.split('_'))) for label in self.table['label'].tolist()]
		if self.load == 'text':
			Y.sort(key=len,reverse=True)

		# Bucketing, X & X_len is dummy when load == 'text'
		self.T = []
		self.X = []
		self.Y = []
		batch_t, batch_x, batch_len, batch_y = [], [], [], []

		for t, x, x_len, y in zip(T, X, X_lens, Y):
			batch_t.append(t)
			batch_x.append(x)
			batch_len.append(x_len)
			batch_y.append(y)
			
			# Fill in batch_x until batch is full
			if len(batch_x) == bucket_size:
				# Half the batch size if seq too long
				if (bucket_size >= 2) and ((max(batch_len) > HALF_BATCHSIZE_TIME) or (max([len(y) for y in batch_y]) > HALF_BATCHSIZE_LABEL)):
					self.T.append(batch_t[:bucket_size//2])
					self.T.append(batch_t[bucket_size//2:])
					self.X.append(batch_x[:bucket_size//2])
					self.X.append(batch_x[bucket_size//2:])
					self.Y.append(batch_y[:bucket_size//2])
					self.Y.append(batch_y[bucket_size//2:])
				else:
					self.T.append(batch_t)
					self.X.append(batch_x)
					self.Y.append(batch_y)
				batch_t, batch_x, batch_len, batch_y = [], [], [], []
		
		# Gather the last batch
		if len(batch_x) > 0:
			self.T.append(batch_t)
			self.X.append(batch_x)
			self.Y.append(batch_y)


#####################
# MEL PHONE DATASET #
#####################
class Mel_Phone_Dataset(LibriDataset):
	
	def __init__(self, file_path, phone_path, sets, bucket_size, max_timestep=0, max_label_len=0, drop=False, load='all'):
		super(Mel_Phone_Dataset, self).__init__(file_path, sets, bucket_size, max_timestep, max_label_len, drop, load)

		assert(self.load == 'phone'), 'This dataset loads mel features and phone boundary labels.'
		# TODO


##################
# GET DATALOADER #
##################
def get_Dataloader(split, load, data_path, batch_size, max_timestep, max_label_len, 
				   use_gpu, n_jobs, dataset, train_set, dev_set, test_set, dev_batch_size, 
				   target_path=None, phone_path=None, decode_beam_size=None, **kwargs):
	if split == 'train':
		bs = batch_size
		shuffle = True
		sets = train_set
		drop_too_long = True
	elif split == 'dev':
		bs = dev_batch_size
		shuffle = False
		sets = dev_set
		drop_too_long = True
	elif split == 'test':
		bs = 1 if decode_beam_size is not None else dev_batch_size
		n_jobs = 1
		shuffle = False
		sets = test_set
		drop_too_long = False
	elif split == 'text':
		bs = batch_size
		shuffle = True
		sets = train_set
		drop_too_long = True
	else:
		raise NotImplementedError('Unsupported `split` argument: ' + split)
		
	if dataset.upper() == "LIBRISPEECH":
		if load in ['all', 'text', 'spec']:
			ds = MelDataset(file_path=data_path, sets=sets, max_timestep=max_timestep, load=load,
						    max_label_len=max_label_len, bucket_size=bs, drop=drop_too_long)
		elif load == 'duo':
			assert(target_path is not None), '`target path` must be provided for this dataset.'
			ds = Mel_Linear_Dataset(file_path=data_path, target_path=target_path, sets=sets, max_timestep=max_timestep, load=load,
						   			max_label_len=max_label_len, bucket_size=bs, drop=drop_too_long)
		elif load == 'phone':
			assert(phone_path is not None), '`phone path` must be provided for this dataset.'
			ds = Mel_Phone_Dataset(file_path=data_path, phone_path=phone_path, sets=sets, max_timestep=max_timestep, load=load,
						   			max_label_len=max_label_len, bucket_size=bs, drop=drop_too_long)
		else:
			raise NotImplementedError('Unsupported `load` argument: ' + load)
	else:
		raise ValueError('Unsupported Dataset: ' + dataset)

	return DataLoader(ds, batch_size=1, shuffle=shuffle, drop_last=False, num_workers=n_jobs, pin_memory=use_gpu)

