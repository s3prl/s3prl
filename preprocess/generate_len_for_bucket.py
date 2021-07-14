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
import sys
import pickle
import argparse
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed
from datasets import load_dataset, Dataset
import soundfile as sf


# change these to match your dataset
# SETS = ['train', 'dev', 'test']
# SETS = ['TRAIN', 'TEST']
SETS = ['train-clean-100', 'train-clean-360', 'train-other-500', 'dev-clean', 'dev-other', 'test-clean', 'test-other']


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

    parser.add_argument('-o', '--output_path', default='./data/', type=str, help='Path to store output', required=False)
    parser.add_argument('-n', '--name', default='len_for_bucket', type=str, help='Name of the output directory', required=False)
    parser.add_argument('--n_jobs', default=-1, type=int, help='Number of jobs used for feature extraction', required=False)

    args = parser.parse_args()
    return args


##################
# EXTRACT LENGTH #
##################
def extract_length(input_file):
    wav, _ = torchaudio.load(input_file)
    return wav.size(-1)


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



#################################
# GENERATE LENGTH WITH DATASETS #
#################################
def compute_audio_length(batch):
    speech_array, _ = sf.read(batch["file"])
    batch["length"] = speech_array.shape[-1]
    return batch


def generate_length_with_datasets(split:str, dataset: Dataset, args):
    dataset = dataset.map(compute_audio_length, num_proc=args.n_jobs)
    dataset = dataset.sort("length", reverse=True)
    dataset = dataset.rename_columns({"file":"file_path"})
    dataset = dataset.add_column("label", [None] * len(dataset))
    output_dir = Path(args.output_path).joinpath(args.name)
    dataset.to_csv(f"{output_dir/split}.csv", columns=["file_path", "length", "label"])


########
# MAIN #
########
def main():

    args = get_preprocess_args()
    asr = load_dataset("superb", "asr")

    for split, dataset in asr.items():
        generate_length_with_datasets(split, dataset, args)


if __name__ == '__main__':
    main()