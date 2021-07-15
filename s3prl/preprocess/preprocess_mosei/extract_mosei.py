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
from utility.asr import encode_target
from utility.audio import extract_feature, mel_dim, num_freq


def boolean_string(s):
    if s not in ['False', 'True']:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--flac_path', default='../../data/mosei/flac', type=str, help='Path to MOSEI segmented FLAC files')
    parser.add_argument('--output_path', default='../../data/mosei', type=str, help='Path to store segmented npys', required=False)
    parser.add_argument('--feature_type', default='mel', type=str, help='Feature type ( mfcc / fbank / mel / linear )', required=False)
    parser.add_argument('--apply_cmvn', default=True, type=boolean_string, help='Apply CMVN on feature', required=False)
    parser.add_argument('--n_jobs', default=-1, type=int, help='Number of jobs used for feature extraction', required=False)
    args = parser.parse_args()
    return args


def extract_mosei(args, dim):
    assert os.path.exists(args.flac_path), f'{args.flac_path} not exists'
    todo = list(Path(args.flac_path).glob("*.flac"))
    print(len(todo),'audio files found in MOSEI')

    assert args.feature_type in ['mel', 'linear', 'fbank'], 'Feature type unsupported'

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    npy_dir = os.path.join(args.output_path,str(args.feature_type)+str(dim))
    for target_dir in [npy_dir]:
        if os.path.exists(target_dir):
            decision = input(f'{target_dir} already exists. Remove it? [Y/N]: ')
            if decision.upper() == 'Y':
                print(f'Removing {target_dir}')
                shutil.rmtree(target_dir)
            else:
                print('Abort')
                exit(0)
        os.makedirs(target_dir)

    print('Extracting acoustic feature...',flush=True)
    tr_x = Parallel(n_jobs=args.n_jobs)(delayed(extract_feature)(str(file), feature=args.feature_type, cmvn=args.apply_cmvn, \
                                        save_feature=os.path.join(npy_dir, str(file).split('/')[-1].replace('.flac',''))) for file in tqdm(todo))


########
# MAIN #
########
def main():
    # get arguments
    args = get_preprocess_args()
    dim = num_freq if args.feature_type == 'linear' else mel_dim

    # Acoustic Feature Extraction
    extract_mosei(args, dim)


if __name__ == '__main__':
    main()

