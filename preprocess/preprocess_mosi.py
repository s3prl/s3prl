# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ preprocess_mosi.py ]
#   Synopsis     [ preprocessing for the MOSI dataset ]
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
from joblib import Parallel, delayed
from utility.asr import encode_target
from utility.audio import extract_feature, mel_dim, num_freq

# using CMU SDK to access audio label
# for installing mmsdk package, please follow the instructions in:
# https://github.com/A2Zadeh/CMU-MultimodalSDK#installation
import mmsdk
from mmsdk import mmdatasdk as md


##################
# BOOLEAB STRING #
##################
def boolean_string(s):
    if s not in ['False', 'True']:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def sdk2npy(string):
    # not data type conversion
    # convert the name format in CMU sdk to corresponding npy file name
    split = string.split('[')
    utterance_name = split[0]
    number = int(split[1].split(']')[0])
    string = utterance_name + '_' + str(number + 1) + '.npy'
    return string

def npy2sdk(string):
    # not data type conversion
    # inverse operation of sdk2npy
    split = string.split('_')
    number = int(split[-1][:-4])
    utterance_name = '_'.join(split[:-1])
    string = utterance_name + '[' + str(number - 1) + ']'
    return string


#############################
# PREPROCESS CONFIGURATIONS #
#############################
def get_preprocess_args():
    parser = argparse.ArgumentParser(description='preprocess arguments for LibriSpeech dataset.')
    parser.add_argument('--data_path', default='/home/leo/d/datasets/MOSI/Raw/Audio/WAV_16000/Segmented', type=str, help='Path to raw MOSI segmented audio dataset')
    parser.add_argument('--output_path', default='../data/', type=str, help='Path to store output', required=False)
    parser.add_argument('--feature_type', default='mel', type=str, help='Feature type ( mfcc / fbank / mel / linear )', required=False)
    parser.add_argument('--apply_cmvn', default=True, type=boolean_string, help='Apply CMVN on feature', required=False)
    parser.add_argument('--n_jobs', default=-1, type=int, help='Number of jobs used for feature extraction', required=False)
    parser.add_argument('--n_tokens', default=5000, type=int, help='Vocabulary size of target', required=False)
    args = parser.parse_args()
    return args


#######################
# ACOUSTIC PREPROCESS #
#######################
def acoustic_preprocess(args, dim):
    # Extracting features from audio
    todo = list(Path(args.data_path).glob("*.wav"))
    print(len(todo),'audio files found in MOSI')

    assert args.feature_type in ['mel', 'linear', 'fbank'], 'Feature type unsupported'
    output_dir = os.path.join(args.output_path,'_'.join(['mosi',str(args.feature_type)+str(dim)]))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print('Extracting acoustic feature...',flush=True)
    tr_x = Parallel(n_jobs=args.n_jobs)(delayed(extract_feature)(str(file), feature=args.feature_type, cmvn=args.apply_cmvn, \
                                        save_feature=os.path.join(output_dir, str(file).split('/')[-1].replace('.wav',''))) for file in tqdm(todo))

    # Loading labels
    DATASET = md.cmu_mosi
    try:
        md.mmdataset(DATASET.labels, args.data_path)
    except RuntimeError:
        print("Labels have been downloaded previously.")

    label_field = 'CMU_MOSI_Opinion_Labels'
    features = [
        label_field,
    ]

    recipe = {feat: os.path.join(args.data_path, feat) + '.csd' for feat in features}
    dataset = md.mmdataset(recipe)
    dataset.align(label_field)

    # Check each label has corresponding utterance
    utterances = os.listdir(output_dir)
    for segment_sdk in dataset[label_field].keys():
        segment_npy = sdk2npy(segment_sdk)
        try:
            assert segment_npy in utterances
        except AssertionError:
            print('AssertionError: Cannot find corresponding utterance for given label')

        # Check the inverse is itself by npy2sdk
        try:
            assert npy2sdk(segment_npy) == segment_sdk
        except AssertionError:
            print('AssertionError: npt2sdk funtion has bug')

    # sort by len
    sorted_xlen = []
    sorted_y = []
    sorted_todo = []
    for idx in reversed(np.argsort(tr_x)):
        filename = str(todo[idx]).split('/')[-1].replace('.wav','.npy')
        sdkname = npy2sdk(filename)
        if sdkname in dataset[label_field].keys():
            sorted_xlen.append(tr_x[idx])
            sorted_y.append(dataset[label_field][sdkname]['features'].reshape(-1)[0])
            sorted_todo.append(filename)

    # Organize label
    df = pd.DataFrame(data={'file_path':[fp for fp in sorted_todo],'length':list(sorted_xlen),'label':sorted_y})

    # Split data into train/dev/test
    train_split = DATASET.standard_folds.standard_train_fold
    dev_split = DATASET.standard_folds.standard_valid_fold
    test_split = DATASET.standard_folds.standard_test_fold

    npy_dir = os.path.join(output_dir, 'npy')
    if not os.path.exists(npy_dir):
        os.mkdir(npy_dir)

    def classify(file_name):
        file_name = file_name[0]
        prefix = '_'.join(file_name.split('_')[:-1])
        shutil.move(os.path.join(output_dir,file_name), os.path.join(npy_dir, file_name))
        if prefix in train_split:
            return 'train'
        elif prefix in dev_split:
            return 'dev'
        elif prefix in test_split:
            return 'test'
        else: assert 0, 'Error in preprocess_mosi.py:146'
    
    belong = np.apply_along_axis(lambda file_name: classify(file_name), 1, df['file_path'].values.reshape(-1, 1))
    df.insert(len(df.columns), 'set', belong)
    train_frame = df[df.set == 'train']
    dev_frame = df[df.set == 'dev']
    test_frame = df[df.set == 'test']
    
    # dump csv of all splits
    df.to_csv(os.path.join(output_dir,'all.csv'))
    train_frame.to_csv(os.path.join(output_dir,'train.csv'))
    dev_frame.to_csv(os.path.join(output_dir,'dev.csv'))
    test_frame.to_csv(os.path.join(output_dir,'test.csv'))

    remain_npy = glob.glob(os.path.join(output_dir, '*.npy'))
    print('Delete ' + str(len(remain_npy)) + ' unlabeled npy files:')
    for npy in remain_npy:
        print('delete ' + npy)
        os.remove(npy)
    print('All done, saved at', output_dir, 'exit.')


########
# MAIN #
########
def main():
    # get arguments
    args = get_preprocess_args()
    dim = num_freq if args.feature_type == 'linear' else mel_dim

    # Acoustic Feature Extraction
    acoustic_preprocess(args, dim)


if __name__ == '__main__':
    main()

    
