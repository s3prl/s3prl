# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ preprocess.py ]
#   Synopsis     [ preprocess text transcripts and audio speech for the LibriSpeech dataset ]
#   Author       [ Andy T. Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
#   Reference    [ https://github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed
from utils.asr import encode_target
from utils.audio import extract_feature, mel_dim, num_freq


##################
# BOOLEAB STRING #
##################
def boolean_string(s):
	if s not in ['False', 'True']:
		raise ValueError('Not a valid boolean string')
	return s == 'True'


#############################
# PREPROCESS CONFIGURATIONS #
#############################
def get_preprocess_args():
	
	parser = argparse.ArgumentParser(description='preprocess arguments for LibriSpeech dataset.')

	parser.add_argument('--data_path', default='./data/', type=str, help='Path to raw LibriSpeech dataset')
	parser.add_argument('--output_path', default='./data/', type=str, help='Path to store output', required=False)

	parser.add_argument('--feature_type', default='fbank', type=str, help='Feature type ( mfcc / fbank / mel / linear )', required=False)
	parser.add_argument('--apply_cmvn', default=True, type=boolean_string, help='Apply CMVN on feature', required=False)

	parser.add_argument('--n_jobs', default=-1, type=int, help='Number of jobs used for feature extraction', required=False)
	parser.add_argument('--n_tokens', default=5000, type=int, help='Vocabulary size of target', required=False)
	parser.add_argument('--target', default='subword', type=str, help='Learning target ( phoneme / char / subword / word )', required=False)

	args = parser.parse_args()
	return args

#######################
# ACOUSTIC PREPROCESS #
#######################
def acoustic_preprocess(args, tr_set, encode_table, output_dir, bpe_dir, dim):
	
	for s in tr_set:
		print('')
		print('Preprocessing',s,'data...',end='')
		todo = list(Path(os.path.join(args.data_path,s)).rglob("*.flac"))
		print(len(todo),'audio files found in',s)

		print('Encoding target...',flush=True)
		if args.target == 'subword':
			tr_y = []
			with open(os.path.join(bpe_dir,'encode', s+'.txt'), 'r') as f:
				for line in f:tr_y.append(line[:-1].split(' '))
		else:
			tr_y = Parallel(n_jobs=args.n_jobs)(delayed(read_text)(str(file),target=args.target) for file in tqdm(todo))
		tr_y, encode_table = encode_target(tr_y, table=encode_table, mode=args.target, max_idx=args.n_tokens)

		if output_dir is None:
			output_dir = os.path.join(args.output_path,'_'.join(['libri',str(args.feature_type)+str(dim),str(args.target)+str(len(encode_table))]))
		if not os.path.exists(output_dir): os.makedirs(output_dir)
		cur_path = os.path.join(output_dir,s)
		if not os.path.exists(cur_path): os.makedirs(cur_path)

		print('Extracting acoustic feature...',flush=True)
		tr_x = Parallel(n_jobs=args.n_jobs)(delayed(extract_feature)(str(file), feature=args.feature_type, cmvn=args.apply_cmvn, \
											save_feature=os.path.join(cur_path, str(file).split('/')[-1].replace('.flac',''))) for file in tqdm(todo))

		# sort by len
		sorted_y = ['_'.join([str(i) for i in tr_y[idx]]) for idx in reversed(np.argsort(tr_x))]
		sorted_todo = [os.path.join(s,str(todo[idx]).split('/')[-1].replace('.flac','.npy')) for idx in reversed(np.argsort(tr_x))]
		# Dump label
		df = pd.DataFrame(data={'file_path':[fp for fp in sorted_todo],'length':list(reversed(sorted(tr_x))),'label':sorted_y})
		df.to_csv(os.path.join(output_dir,s+'.csv'))

		with open(os.path.join(output_dir,"mapping.pkl"), "wb") as fp:
			pickle.dump(encode_table, fp)

	print('All done, saved at', output_dir, 'exit.')


########
# MAIN #
########
def main():

	# get arguments
	args = get_preprocess_args()

	# Process data
	sets = ['train-clean-100','train-clean-360','train-other-500','dev-clean','dev-other','test-clean','test-other']
	encode_table = None
	output_dir = None
	dim = num_freq if args.feature_type == 'linear' else mel_dim

	# BPE training
	if args.target == 'subword':
		encode_table, output_dir, bpe_dir = subword_preprocess(args, sets, encode_table, output_dir, dim)

	# Select data sets
	print('')
	print('Data sets :')
	for idx,s in enumerate(sets):
		print('\t',idx,':',s)
	tr_set = input('Please enter the index of splits you wish to use preprocess. (seperate with space): ')
	tr_set = [sets[int(t)] for t in tr_set.split(' ')]

	# Acoustic Feature Extraction & Make Date Table
	acoustic_preprocess(args, tr_set, encode_table, output_dir, bpe_dir, dim)


if __name__ == '__main__':
	main()

	
