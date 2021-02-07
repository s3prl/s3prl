# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ preprocess_any.py ]
#   Synopsis     [ preprocess text transcripts and audio speech for any dataset ]
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
from utility.audio import extract_feature, num_mels, num_mfcc, num_freq


SETS = ['train', 'dev', 'test'] # change these to match your dataset
# SETS = ['train-clean-100', 'train-clean-360', 'train-other-500', 'dev-clean', 'dev-other', 'test-clean', 'test-other']#


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
    
    parser = argparse.ArgumentParser(description='preprocess arguments for any dataset.')

    parser.add_argument('--output_path', default='./data/', type=str, help='Path to store output', required=False)
    parser.add_argument('--audio_extention', default='.flac', type=str, help='audio file type (.wav / .flac / .mp3 / etc)', required=False)

    parser.add_argument('--feature_type', default='fbank', type=str, help='Feature type ( mfcc / fbank / mel / linear )', required=False)
    parser.add_argument('--delta', default=False, type=boolean_string, help='Append Delta', required=False)
    parser.add_argument('--delta_delta', default=False, type=boolean_string, help='Append Delta Delta', required=False)
    parser.add_argument('--apply_cmvn', default=True, type=boolean_string, help='Apply CMVN on feature', required=False)

    parser.add_argument('--n_jobs', default=-1, type=int, help='Number of jobs used for feature extraction', required=False)
    parser.add_argument('--name', default='None', type=str, help='Name of the output directory', required=False)

    args = parser.parse_args()
    return args


#######################
# ACOUSTIC PREPROCESS #
#######################
def acoustic_preprocess(args, tr_set, dim, audio_extention):
    
    for i, s in enumerate(tr_set):
        print('')
        print('Preprocessing data in: ', s, end='')
        todo = list(Path(os.path.join(args.data_root, s)).rglob('*' + audio_extention)) # '*.flac'
        print(len(todo), 'audio files found.')

        if args.name == 'None':
            output_dir = os.path.join(args.output_path, '_'.join(['NewData', str(args.feature_type)+str(dim)]))
        else:
            output_dir = os.path.join(args.output_path, args.name)
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        cur_path = os.path.join(output_dir, tr_set[i])
        if not os.path.exists(cur_path): os.makedirs(cur_path)

        print('Extracting acoustic feature...', flush=True)
        tr_x = Parallel(n_jobs=args.n_jobs)(delayed(extract_feature)(str(file), feature=args.feature_type, \
                                            delta=args.delta, delta_delta=args.delta_delta, cmvn=args.apply_cmvn, \
                                            save_feature=os.path.join(cur_path, str(file).split('/')[-1].replace(audio_extention, ''))) for file in tqdm(todo))

        # sort by len
        sorted_todo = [os.path.join(tr_set[i], str(todo[idx]).split('/')[-1].replace(audio_extention, '.npy')) for idx in reversed(np.argsort(tr_x))]
        # Dump data
        df = pd.DataFrame(data={'file_path':[fp for fp in sorted_todo], 'length':list(reversed(sorted(tr_x))), 'label':None})
        df.to_csv(os.path.join(output_dir, tr_set[i] + '.csv'))

    print('All done, saved at', output_dir, 'exit.')


########
# MAIN #
########
def main():

    # get arguments
    args = get_preprocess_args()
    mel_dim = num_mels * (1 + int(args.delta) + int(args.delta_delta))
    mfcc_dim = num_mfcc * (1 + int(args.delta) + int(args.delta_delta))
    dim = num_freq if args.feature_type == 'linear' else (mfcc_dim if args.feature_type == 'mfcc' else mel_dim)
    print('Delta: ', args.delta, '. Delta Delta: ', args.delta_delta, '. Cmvn: ', args.apply_cmvn)

    # Select data sets
    for idx, s in enumerate(SETS):
        print('\t', idx, ':', s)
    tr_set = input('Please enter the index of splits you wish to use preprocess. (seperate with space): ')
    tr_set = [SETS[int(t)] for t in tr_set.split(' ')]

    # Acoustic Feature Extraction & Make Data Table
    acoustic_preprocess(args, tr_set, dim, args.audio_extention)


if __name__ == '__main__':
    main()