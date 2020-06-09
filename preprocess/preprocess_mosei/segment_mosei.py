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
from pydub import AudioSegment


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
    parser.add_argument('--output_path', default='../../data/mosei', type=str, help='Path to store segmented flac and npys. Should already contains mosei_no_semi.csv', required=False)
    args = parser.parse_args()
    return args


def segment_mosei(args):
    output_dir = args.output_path
    mosei_summary = os.path.join(output_dir, 'mosei_no_semi.csv')
    flac_dir = os.path.join(output_dir, 'flac')
    assert os.path.exists(mosei_summary), 'Output path should already be created with a mosei_no_semi.csv inside it'
    for target_dir in [flac_dir]:
        if os.path.exists(target_dir):
            decision = input(f'{target_dir} already exists. Remove it? [Y/N]: ')
            if decision.upper() == 'Y':
                shutil.rmtree(target_dir)
                print(f'{target_dir} removed')
            else:
                print('Abort')
                exit(0)
        os.makedirs(target_dir)

    df = pd.read_csv(mosei_summary)

    for index, row in df.iterrows():
        underscore = row.key
        wavname = f'{row.filename}.wav'
        wavpath = os.path.join(args.data_path, wavname)
        assert os.path.exists(wavpath), f'wav not exists: {wavpath}'
        wav = AudioSegment.from_wav(wavpath)

        start = int(row.start * 1000)
        end = int(row.end * 1000)
        assert start >= 0, f'{underscore} has negative start time'
        assert end >= 0, f'{underscore} has negative end time'
        seg_wav = wav[start:end]
        seg_flacpath = os.path.join(flac_dir, f'{underscore}.flac')
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

