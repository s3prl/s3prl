# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ observe_speaker.py ]
#   Synopsis     [ Analyze the speaker training set for LibriSpeech ]
#   Author       [ Andy T. Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset

############
# SETTINGS #
############
SPEAKER_THRESHOLD = 120
root = '../data/libri_mel160'
drop = True
max_timestep = 3000
max_label_len = 400


def get_speaker_from_path(x):
    return x.split('/')[-1].split('.')[0].split('-')[0]


def get_all_speakers(X):
    speaker_set = {}
    for x in X:
        speaker = get_speaker_from_path(x)
        if speaker not in speaker_set:
            speaker_set[speaker] = 0
        else:
            speaker_set[speaker] += 1
    return speaker_set


def compute_speaker2idx(speakers):
    idx = 0
    speaker2idx = {}
    for speaker in sorted(speakers):
        if speaker not in speaker2idx and speakers[speaker] > SPEAKER_THRESHOLD: # eliminate the speakers with too few utterance
            speaker2idx[speaker] = idx
            idx += 1
    return speaker2idx


########
# MAIN #
########
def main():

    # Load the train-clean-100 set
    tables = pd.read_csv(os.path.join(root, 'train-clean-100' + '.csv'))

    # Compute speaker dictionary
    print('[Dataset] - Computing speaker class...')
    O = tables['file_path'].tolist()
    speakers = get_all_speakers(O)
    speaker2idx = compute_speaker2idx(speakers)
    class_num = len(speaker2idx)
    print('[Dataset] - Possible speaker classes: ', class_num)
    

    train = tables.sample(frac=0.9, random_state=20190929) # random state is a seed value
    test = tables.drop(train.index)
    table = train.sort_values(by=['length'], ascending=False)

    X = table['file_path'].tolist()
    X_lens = table['length'].tolist()

    # Crop seqs that are too long
    if drop and max_timestep > 0:
        table = table[table.length < max_timestep]
    if drop and max_label_len > 0:
        table = table[table.label.str.count('_')+1 < max_label_len]

    # computer utterance per speaker
    num_utt = []
    for speaker in speakers:
        if speaker in speaker2idx:
            num_utt.append(speakers[speaker])
    print('Average utterance per speaker: ', np.mean(num_utt))

    # TODO: furthur analysis


if __name__ == '__main__':
    main()