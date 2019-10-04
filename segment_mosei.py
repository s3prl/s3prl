# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ preprocess_mosei.py ]
#   Synopsis     [ preprocessing for the MOSEI dataset ]
#   Author       [ Andy T. Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import shutil
import glob
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import shutil
from joblib import Parallel, delayed
from utils.asr import encode_target
from utils.audio import extract_feature, mel_dim, num_freq
from pydub import AudioSegment

# using CMU SDK to access audio label
# for installing mmsdk package, please follow the instructions in:
# https://github.com/A2Zadeh/CMU-MultimodalSDK#installation
import mmsdk
from mmsdk import mmdatasdk as md


def bracket_underscore(string):
	split = string.split('[')
	utterance_name = split[0]
	number = int(split[1].split(']')[0])
	string = utterance_name + '_' + str(number + 1)
	return string

def underscore_bracket(string):
	split = string.split('_')
	number = int(split[-1][:-4])
	utterance_name = '_'.join(split[:-1])
	string = utterance_name + '[' + str(number - 1) + ']'
	return string


def get_preprocess_args():
	parser = argparse.ArgumentParser(description='preprocess arguments for LibriSpeech dataset.')
	parser.add_argument('--data_path', default='/home/leo/d/datasets/MOSEI/Raw/Audio/Full/WAV_16000', type=str, help='Path to MOSEI non-segmented WAV files')
	parser.add_argument('--output_path', default='./data/mosei_segmented_flac', type=str, help='Path to store segmented flac', required=False)
	args = parser.parse_args()
	return args


def segment_mosei(args):
	flac_dir = args.output_path
	if os.path.exists(flac_dir):
		shutil.rmtree(flac_dir)
	os.makedirs(flac_dir)

	# Loading labels
	DATASET = md.cmu_mosei
	try:
		md.mmdataset(DATASET.labels, args.data_path)
	except RuntimeError:
		print("Labels have been downloaded previously.")

	label = 'CMU_MOSEI_LabelsSentiment'
	features = [
		label,
	]
	recipe = {feat: os.path.join(args.data_path, feat) + '.csd' for feat in features}
	dataset = md.mmdataset(recipe)
	dataset.align(label)

	for key in iter(dataset[label].keys()):
		underscore = bracket_underscore(key)
		underscore_split = underscore.split('_')
		prefix = '_'.join(underscore_split[:-1])
		postfix = underscore_split[-1]
		wavname = f'{prefix}.wav'
		wavpath = os.path.join(args.data_path, wavname)
		if not os.path.exists(wavpath):
			assert False, f'wav not exists: {wavpath}'
		wav = AudioSegment.from_wav(wavpath)

		seg_flacpath = os.path.join(flac_dir, f'{underscore}.flac')
		item = dataset[label][key]
		start = int(item['intervals'][0, 0] * 1000)
		end = int(item['intervals'][0, 1] * 1000)
		seg_wav = wav[start:end]
		seg_wav.export(seg_flacpath, format='flac', parameters=['-ac', '1', '-sample_fmt', 's16', '-ar', '16000'])


########
# MAIN #
########
def main():
	# get arguments
	args = get_preprocess_args()

	# Acoustic Feature Extraction
	segment_mosei(args)


if __name__ == '__main__':
	main()

	
