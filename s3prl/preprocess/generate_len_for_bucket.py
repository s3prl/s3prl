# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ generate_len_for_bucket.py ]
#   Synopsis     [ preprocess audio speech to generate meta data for dataloader bucketing ]
#   Author       [ Andy T. Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
#   Reference    [ https://github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import argparse
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed


#############################
# PREPROCESS CONFIGURATIONS #
#############################
def get_preprocess_args():
    
    parser = argparse.ArgumentParser(description='preprocess arguments for any dataset.')

    parser.add_argument('-i', '--input_data', default='../LibriSpeech/', type=str, help='Path to your LibriSpeech directory', required=False)
    parser.add_argument('-o', '--output_path', default='./data/', type=str, help='Path to store output', required=False)
    parser.add_argument('-a', '--audio_extension', default='.flac', type=str, help='audio file type (.wav / .flac / .mp3 / etc)', required=False)
    parser.add_argument('-n', '--name', default='len_for_bucket', type=str, help='Name of the output directory', required=False)
    parser.add_argument('--n_jobs', default=-1, type=int, help='Number of jobs used for feature extraction', required=False)

    args = parser.parse_args()
    return args


##################
# EXTRACT LENGTH #
##################
def extract_length(input_file):
    torchaudio.set_audio_backend("sox_io")
    return torchaudio.info(input_file).num_frames


###################
# GENERATE LENGTH #
###################
def generate_length(args, tr_set, audio_extension):
    
    for i, s in enumerate(tr_set):
        if os.path.isdir(os.path.join(args.input_data, s.lower())):
            s = s.lower()
        elif os.path.isdir(os.path.join(args.input_data, s.upper())):
            s = s.upper()
        else:
            assert NotImplementedError

        print('')
        todo = list(Path(os.path.join(args.input_data, s)).rglob('*' + audio_extension)) # '*.flac'
        print(f'Preprocessing data in: {s}, {len(todo)} audio files found.')

        output_dir = os.path.join(args.output_path, args.name)
        if not os.path.exists(output_dir): os.makedirs(output_dir)

        print('Extracting audio length...', flush=True)
        tr_x = Parallel(n_jobs=args.n_jobs)(delayed(extract_length)(str(file)) for file in tqdm(todo))

        # sort by len
        sorted_todo = [os.path.join(s, str(todo[idx]).split(s+'/')[-1]) for idx in reversed(np.argsort(tr_x))]
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
    
    if 'librilight' in args.input_data.lower():
        SETS = ['small', 'medium', 'large'] + ['small-splitted', 'medium-splitted', 'large-splitted']
    elif 'librispeech' in args.input_data.lower():
        SETS = ['train-clean-100', 'train-clean-360', 'train-other-500', 'dev-clean', 'dev-other', 'test-clean', 'test-other']
    elif 'timit' in args.input_data.lower():
        SETS = ['TRAIN', 'TEST']
    else:
        raise NotImplementedError
    # change the SETS list to match your dataset, for example:
    # SETS = ['train', 'dev', 'test']
    # SETS = ['TRAIN', 'TEST']
    # SETS = ['train-clean-100', 'train-clean-360', 'train-other-500', 'dev-clean', 'dev-other', 'test-clean', 'test-other']
    
    # Select data sets
    for idx, s in enumerate(SETS):
        print('\t', idx, ':', s)
    tr_set = input('Please enter the index of splits you wish to use preprocess. (seperate with space): ')
    tr_set = [SETS[int(t)] for t in tr_set.split(' ')]

    # Acoustic Feature Extraction & Make Data Table
    generate_length(args, tr_set, args.audio_extension)


if __name__ == '__main__':
    main()
