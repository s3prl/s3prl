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
class LibriDataset(Dataset):
	def __init__(self, file_path, sets, bucket_size, max_timestep=0, max_label_len=0, drop=False, load='all'):
		# Read file
		self.root = file_path
		tables = [pd.read_csv(os.path.join(file_path,s+'.csv')) for s in sets]
		self.table = pd.concat(tables,ignore_index=True).sort_values(by=['length'],ascending=False)
		self.load = load
		assert self.load in ['all', 'text', 'spec']

		# Crop seqs that are too long
		if drop and max_timestep > 0 and self.load != 'text':
			self.table = self.table[self.table.length < max_timestep]
		if drop and max_label_len > 0:
			self.table = self.table[self.table.label.str.count('_')+1 < max_label_len]

		X = self.table['file_path'].tolist()
		X_lens = self.table['length'].tolist()
			
		Y = [list(map(int, label.split('_'))) for label in self.table['label'].tolist()]
		if self.load == 'text':
			Y.sort(key=len,reverse=True)

		# Bucketing, X & X_len is dummy when load == 'text'
		self.X = []
		self.Y = []
		tmp_x,tmp_len,tmp_y = [],[],[]

		for x,x_len,y in zip(X,X_lens,Y):
			tmp_x.append(x)
			tmp_len.append(x_len)
			tmp_y.append(y)
			# Half the batch size if seq too long
			if len(tmp_x)== bucket_size:
				if (bucket_size>=2) and ((max(tmp_len)> HALF_BATCHSIZE_TIME) or (max([len(y) for y in tmp_y])>HALF_BATCHSIZE_LABEL)):
					self.X.append(tmp_x[:bucket_size//2])
					self.X.append(tmp_x[bucket_size//2:])
					self.Y.append(tmp_y[:bucket_size//2])
					self.Y.append(tmp_y[bucket_size//2:])
				else:
					self.X.append(tmp_x)
					self.Y.append(tmp_y)
				tmp_x,tmp_len,tmp_y = [],[],[]
		if len(tmp_x)>0:
			self.X.append(tmp_x)
			self.Y.append(tmp_y)


	def __getitem__(self, index):
		# Load label
		if self.load != 'spec':
			y = [y for y in self.Y[index]]
			y = target_padding(y, max([len(v) for v in y]))
			if self.load == 'text':
				return y
		
		# Load acoustic feature and pad
		x = [torch.FloatTensor(np.load(os.path.join(self.root,f))) for f in self.X[index]]
		x = pad_sequence(x, batch_first=True)
		if self.load == 'spec':
			return x
		else return x,y
			
	
	def __len__(self):
		return len(self.Y)


################
# LOAD DATASET #
################
def LoadDataset(split, load, data_path, batch_size, max_timestep, max_label_len, use_gpu, n_jobs,
				dataset, train_set, dev_set, test_set, dev_batch_size, decode_beam_size, **kwargs):
	if split=='train':
		bs = batch_size
		shuffle = True
		sets = train_set
		drop_too_long = True
	elif split=='dev':
		bs = dev_batch_size
		shuffle = False
		sets = dev_set
		drop_too_long = True
	elif split=='test':
		bs = 1 if decode_beam_size>1 else dev_batch_size
		n_jobs = 1
		shuffle = False
		sets = test_set
		drop_too_long = False
	elif split=='text':
		bs = batch_size
		shuffle = True
		sets = train_set
		drop_too_long = True
	else:
		raise NotImplementedError
		
	if dataset.upper() =="LIBRISPEECH":
		ds = LibriDataset(file_path=data_path, sets=sets, max_timestep=max_timestep, load=load,
						   max_label_len=max_label_len, bucket_size=bs, drop=drop_too_long)
	else:
		raise ValueError('Unsupported Dataset: '+dataset)

	return DataLoader(ds, batch_size=1, shuffle=shuffle, drop_last=False, num_workers=n_jobs, pin_memory=use_gpu)


