# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ preprocess_timit.py ]
#   Synopsis     [ preprocess text transcripts and audio speech for the TIMIT dataset ]
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
from utility.asr import encode_target
from utility.audio import extract_feature, mel_dim, num_freq


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

    parser.add_argument('--data_path', default='./data/timit', type=str, help='Path to raw TIMIT dataset')
    parser.add_argument('--output_path', default='./data/', type=str, help='Path to store output', required=False)

    parser.add_argument('--feature_type', default='mel', type=str, help='Feature type ( mfcc / fbank / mel / linear )', required=False)
    parser.add_argument('--apply_cmvn', default=True, type=boolean_string, help='Apply CMVN on feature', required=False)

    parser.add_argument('--n_jobs', default=-1, type=int, help='Number of jobs used for feature extraction', required=False)
    parser.add_argument('--n_tokens', default=1000, type=int, help='Vocabulary size of target', required=False)
    parser.add_argument('--target', default='phoneme', type=str, help='Learning target ( phoneme / char / subword / word )', required=False)

    args = parser.parse_args()
    return args


#############
# READ TEXT #
#############
def read_text(file, target):
    labels = []
    if target == 'phoneme':
        with open(file.replace('.wav','.phn'),'r') as f:
            for line in f:
                labels.append(line.replace('\n','').split(' ')[-1])
    elif target in ['char','subword','word']:
        with open(file.replace('.wav','.wrd'),'r') as f:
            for line in f:
                labels.append(line.replace('\n','').split(' ')[-1])
        if target =='char':
            labels = [c for c in ' '.join(labels)]
    else:
        raise ValueError('Unsupported target: ' + target)
    return labels


####################
# PREPROCESS TRAIN #
####################
def preprocess_train(args, dim):
    # Process training data
    print('')
    print('Preprocessing training data...', end='')
    todo = list(Path(os.path.join(args.data_path, 'TRAIN')).rglob("*.[wW][aA][vV]"))
    if len(todo) == 0: todo = list(Path(os.path.join(args.data_path, 'train')).rglob("*.[wW][aA][vV]"))
    print(len(todo), 'audio files found in training set (should be 4620)')


    print('Extracting acoustic feature...', flush=True)
    tr_x = Parallel(n_jobs=args.n_jobs)(delayed(extract_feature)(str(file), feature=args.feature_type, cmvn=args.apply_cmvn) for file in tqdm(todo))
    print('Encoding training target...', flush=True)
    tr_y = Parallel(n_jobs=args.n_jobs)(delayed(read_text)(str(file), target=args.target) for file in tqdm(todo))
    tr_y, encode_table = encode_target(tr_y, table=None, mode=args.target, max_idx=args.n_tokens)

    output_dir = os.path.join(args.output_path,'_'.join(['timit', str(args.feature_type) + str(dim), str(args.target) + str(len(encode_table))]))
    print('Saving training data to', output_dir)
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    with open(os.path.join(output_dir, 'train_x.pkl'), 'wb') as fp:
        pickle.dump(tr_x, fp)
    del tr_x
    with open(os.path.join(output_dir, 'train_y.pkl'), 'wb') as fp:
        pickle.dump(tr_y, fp)
    del tr_y
    with open(os.path.join(output_dir, 'mapping.pkl'), 'wb') as fp:
        pickle.dump(encode_table, fp)
    with open(os.path.join(output_dir, 'train_id.pkl'), 'wb') as fp:
        pickle.dump(todo, fp)
    return encode_table, output_dir



###################
# PREPROCESS TEST #
###################
def preprocess_test(args, encode_table, output_dir, dim):
    
    # Process testing data
    print('Preprocessing testing data...', end='')
    todo = list(Path(os.path.join(args.data_path, 'TEST')).rglob("*.[wW][aA][vV]"))
    if len(todo) == 0: todo = list(Path(os.path.join(args.data_path, 'test')).rglob("*.[wW][aA][vV]"))
    print(len(todo), 'audio files found in test set (should be 1680)')

    print('Extracting acoustic feature...', flush=True)
    tt_x = Parallel(n_jobs=args.n_jobs)(delayed(extract_feature)(str(file), feature=args.feature_type, cmvn=args.apply_cmvn) for file in tqdm(todo))
    print('Encoding testing target...', flush=True)
    tt_y = Parallel(n_jobs=args.n_jobs)(delayed(read_text)(str(file), target=args.target) for file in tqdm(todo))
    tt_y, encode_table = encode_target(tt_y, table=encode_table, mode=args.target, max_idx=args.n_tokens)


    print('Saving testing data to',output_dir)
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    with open(os.path.join(output_dir, "test_x.pkl"), "wb") as fp:
        pickle.dump(tt_x, fp)
    del tt_x
    with open(os.path.join(output_dir, "test_y.pkl"), "wb") as fp:
        pickle.dump(tt_y, fp)
    del tt_y
    with open(os.path.join(output_dir, 'test_id.pkl'), 'wb') as fp:
        pickle.dump(todo, fp)


########
# MAIN #
########
def main():

    # get arguments
    args = get_preprocess_args()
    dim = num_freq if args.feature_type == 'linear' else mel_dim

    # Process data
    encode_table, output_dir = preprocess_train(args, dim)
    preprocess_test(args, encode_table, output_dir, dim)
    print('All done, saved at \'' + str(output_dir) + '\' exit.')

if __name__ == '__main__':
    main()

    
