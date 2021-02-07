# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ preprocess_alignment.py ]
#   Synopsis     [ preprocess phone alignment for the LibriSpeech dataset ]
#   Author       [ Andy T. Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
#   Reference    [ https://github.com/BogiHsu/Phone-Recognizer/blob/815cf9375045c053fa57d17fad0fa14fdc3c7bee/loader.py#L28 ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from utility.audio import sample_rate, _stft_parameters


#############################
# PREPROCESS CONFIGURATIONS #
#############################
def get_preprocess_args():
    
    parser = argparse.ArgumentParser(description='preprocess arguments for LibriSpeech dataset.')
    parser.add_argument('--data_path', default='./data/libri_alignment', type=str, help='Path to raw LibriSpeech alignment')
    parser.add_argument('--output_path', default='./data/libri_phone', type=str, help='Path to store output', required=False)
    args = parser.parse_args()
    return args


####################
# PHONE PREPROCESS #
####################
def phone_preprocess(data_path, output_path, sets, unaligned):
    
    print('Data sets :')
    for idx, s in enumerate(sets):
        print('\t', idx, ':', s)
    todo_sets = input('Please enter the index for preprocessing sets (seperate w/ space): ')
    sets = [sets[int(s)] for s in todo_sets.split(' ')]
    
    # compute phone2idx
    idx = 0
    phone2idx = {}
    for s in sets:
        print('')
        print('Computing', s, 'data...')
        for path in tqdm(list(Path(os.path.join(data_path, s)).rglob("*.txt"))):
            check_name = path.as_posix().split('/')[-1].split('.')[0]
            if check_name not in unaligned and check_name != 'unaligned': # ignore the unaligned files and `unaligned.txt` itself
                for line in open(path).readlines():
                    phone = line.strip('\n').split(' ')[-1]
                    if phone not in phone2idx:
                        phone2idx[phone] = idx
                        idx += 1
    print('Phone set:')
    print(phone2idx)
    print(len(phone2idx), 'distinct phones found in', sets)
    with open(os.path.join(output_path, 'phone2idx.pkl'), "wb") as fp:
            pickle.dump(phone2idx, fp)

    for s in sets:
        print('')
        print('Preprocessing', s, 'data...')
        todo = list(Path(os.path.join(data_path, s)).rglob("*.txt"))
        print(len(todo),'audio files found in', s)
        if not os.path.exists(os.path.join(output_path, s)): 
            os.makedirs(os.path.join(output_path, s))

        print('Preprocessing phone alignments...', flush=True)
        for path in tqdm(todo):
            check_name = path.as_posix().split('/')[-1].split('.')[0]
            if check_name not in unaligned and check_name != 'unaligned': # ignore the unaligned files and `unaligned.txt` itself
                x = []
                file = open(path).readlines()
                for line in file:
                    line = line.strip('\n').split(' ')
                    x += time_to_frame(start_time=float(line[0]), end_time=float(line[1]), phone=phone2idx[line[2]])
                x = np.asarray(x)
                path_to_save = str(path).replace(data_path.split('/')[-1], output_path.split('/')[-1]).replace('txt', 'pkl')
                with open(path_to_save, "wb") as fp:
                    pickle.dump(x, fp)

    print('Phone preprocessing complete!')      


#################
# TIME TO FRAME #
#################
def time_to_frame(start_time, end_time, phone):
    phones = []
    
    start_time = int(start_time * sample_rate)
    end_time = int(end_time * sample_rate)
    
    _, hop_length, win_length = _stft_parameters(sample_rate=sample_rate)
    h_window = win_length * 0.5 # select the middle of a window

    start_time = (start_time - h_window) if start_time >= h_window else 0
    end_time = (end_time - h_window) if end_time >= h_window else 0
    times = (end_time // hop_length) - (start_time // hop_length) \
            + (1 if start_time % hop_length == 0 else 0) - (1 if end_time % hop_length == 0 else 0)
    phones += [phone] * int(times)
    return phones


########
# MAIN #
########
def main():

    # get arguments
    args = get_preprocess_args()
    
    # mkdir
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # dump unaligned text
    try:
        file = open(os.path.join(args.data_path, 'train-clean-360/unaligned.txt')).readlines()
        unaligned = [str(line).split('\t')[0].split(' ')[0] for line in file]
        print('Unaligned list: ', unaligned)
        unaligned_pkl = ['train-clean-360/' + u + '.npy' for u in unaligned]
        with open(os.path.join(args.output_path, 'unaligned.pkl'), "wb") as fp:
            pickle.dump(unaligned_pkl, fp)
    except:
        raise ValueError('Did not find unaligned.txt!')

    # Process data
    sets = ['train-clean-360', 'test-clean'] # only two sets available for now
    # sets = ['train-clean-100','train-clean-360','train-other-500','dev-clean','dev-other','test-clean','test-other']
    phone_preprocess(args.data_path, args.output_path, sets, unaligned)


if __name__ == '__main__':
    main()