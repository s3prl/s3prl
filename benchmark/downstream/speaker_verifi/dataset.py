# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ dataset.py ]
#   Synopsis     [ the speaeker_verifi dataset ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import random
#-------------#
import numpy as np
import pandas as pd
#-------------#
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset
#-------------#
import os
import torch
import random
import torchaudio
from tqdm import tqdm
from pathlib import Path
from os.path import join, getsize
from joblib import Parallel, delayed
from torch.utils.data import Dataset
from librosa.util import find_files


# Voxceleb 1 + 2 Speaker Verification
# class Voxceleb_Dataset(Dataset):
#     def __init__(self, file_path, max_timestep=500, meta_data=None, dev_spkr_ids=None, utterance_sample=5, is_dev=False, is_eval=False):

#         # Read file
#         self.root = file_path
#         self.all_speakers  = [f.path for f in os.scandir(file_path) if f.is_dir()]
#         # Crop seqs that are too long
#         self.max_timestep = max_timestep
#         self.utterance_sample = utterance_sample
#         self.is_eval = is_eval
#         self.meta_data =meta_data
#         self.pair_table = None
#         self.is_dev = is_dev

#         if not self.is_dev and not self.is_eval:
#             self.speakers  = [path for path in self.all_speakers if path.split("/")[-1] not in dev_spkr_ids]
#             self.speaker_num = len(self.speakers)
#             self.number_range = list(range(self.speaker_num))
        
#         if self.is_eval:
#             self.pair_table = []
#             usage_list = open(self.meta_data, "r").readlines()
#             for pair in usage_list:
#                 list_pair = pair.split()
#                 pair_1= os.path.join(self.root, list_pair[1])
#                 pair_2= os.path.join(self.root, list_pair[2])
#                 one_pair = [list_pair[0],pair_1,pair_2 ]
#                 self.pair_table.append(one_pair)

#         if self.is_dev:
#             self.pair_table = []
#             usage_list = open(self.meta_data, "r").readlines()
#             for pair in usage_list:
#                 list_pair = pair.split()
#                 pair_1= os.path.join(self.root, list_pair[1])
#                 pair_2= os.path.join(self.root, list_pair[2])
#                 one_pair = [list_pair[0],pair_1,pair_2 ]
#                 self.pair_table.append(one_pair)

            

#     def __len__(self):
#         if self.is_eval or self.is_dev:
#             return len(self.pair_table)
#         else:
#             return len(self.speakers)
    
#     def __getitem__(self, idx):
#         if self.is_eval or self.is_dev:
#             y_label, x1_path, x2_path = self.pair_table[idx]
#             wav1, sr = torchaudio.load(x1_path)
#             wav2, _ = torchaudio.load(x2_path)

#             wav1 = wav1.squeeze(0)
#             wav2 = wav2.squeeze(0)

#             length1 = wav1.shape[0]
#             length2 = wav2.shape[0]

#             return wav1, wav2, torch.tensor(int(y_label[0])), torch.tensor(length1), torch.tensor(length2)
#         else:
#             path = random.sample(find_files(self.speakers[idx],ext=["wav"]), self.utterance_sample)
#             x_list = []
#             length_list = []
#             index = []

#             for i in range(len(path)):
#                 wav, sr=torchaudio.load(path[i])
#                 wav = wav.squeeze(0)
#                 length = wav.shape[0]
#                 if length > self.max_timestep:
#                     x = wav[:self.max_timestep]
#                     length = self.max_timestep
#                 else:
#                     x = wav
#                 x_list.append(x)
#                 index.append(torch.tensor(idx).long())
#                 length_list.append(torch.tensor(length).long())
            
#             return x_list, torch.stack(length_list), torch.stack(index)
        
#     def collate_test_fn(self, data_sample):
#         # wavs: [(wavs[0:utterance_length],lengths[0:utterance_length]), ...]
#         wavs1 = []
#         wavs2 = []
#         ylabels = []
#         lengths1 = []
#         lengths2 = []

#         for samples in data_sample:
#             wavs1.append(samples[0])
#             wavs2.append(samples[1])
#             ylabels.append(samples[2])
#             lengths1.append(samples[3])
#             lengths2.append(samples[4])

#         all_wav = []
#         all_wav.extend(wavs1)
#         all_wav.extend(wavs2)

#         all_length = []
#         all_length.extend(lengths1)
#         all_length.extend(lengths2)

#         # length_tensor = torch.stack(all_length)
#         # back_wavs = pad_sequence(all_wav, batch_first=True)
#         # back_length = length_tensor

#         return all_wavs, all_length, ylabel


#     def collate_train_fn(self,data_sample):
#         # wavs: [(wavs[0:utterance_length],lengths[0:utterance_length]), ...]
#         wavs = []
#         lengths = []
#         indexes = []

#         for samples in data_sample:
#             wavs.extend(samples[0])
#             lengths.extend(samples[1])

#         # length_tensor = torch.stack(lengths)
#         # back_wavs = pad_sequence(wavs, batch_first=True)
#         # back_length = length_tensor

#         return wavs, lengths


# Voxceleb 1 + 2 
# preprocessing need seperate folder to dev, train, test
class SpeakerVerifi_train(Dataset):
    def __init__(self, directory_path,utter_sample=5):

        self.root = directory_path
        self.all_speakers = [f.path for f in os.scandir(self.root) if f.is_dir()]
        self.utter_number = utter_sample
        self.necessary_dict = self.processing()
        self.dataset = self.necessary_dict['spk_paths'] 
        

    def processing(self):
        
        speaker_num = len(self.all_speakers)
        return {"spk_paths":self.all_speaker,"total_spk_num":speaker_num,"pair_table":None}
    
    def __len__(self):
        return self.necessary_dict['total_spk_num']


    def __getitem__(self, idx):
        
        path = random.sample(find_files(self.dataset[idx], ext=['wav']), self.utter_number)

        x_list = []
        length_list = []

        for i in range(len(path)):
            wav, sr = torchaudio.load(path[i])
            wav = wav.squeeze(0)
            length = wav.shape[0]

            x_list.append(wav)
            length_list.append(torch.tensor(length).long())

        return x_list, length_list
    
    def collate_fn(self,data_sample):

        wavs = []
        lengths = []
        indexes = []

        for samples in data_sample:
            wavs.extend(samples[0])
            lengths.extend(samples[1])

        return wavs, lengths


class SpeakerVerifi_test(Dataset):
    def __init__(self, directory_path,meta_data=None):

        self.root = directory_path
        self.meta_data = meta_data
        self.necessary_dict = self.processing()
        self.dataset = self.necessary_dict['pair_table'] if self.necessary_dict['pair_table'] else self.necessary_dict['spk_paths']
        
    def processing(self):
        pair_table = []
        usage_list = open(self.meta_data, "r").readlines()
        for pair in usage_list:
            list_pair = pair.split()
            pair_1= os.path.join(self.root, list_pair[1])
            pair_2= os.path.join(self.root, list_pair[2])
            one_pair = [list_pair[0],pair_1,pair_2 ]
            pair_table.append(one_pair)
        return {"spk_paths":None,"total_spk_num":None,"pair_table":pair_table}

    def __len__(self):
        return len(self.necessary_dict['pair_table'])

    def __getitem__(self, idx):
        y_label, x1_path, x2_path = self.pair_table[idx]
        wav1, _ = torchaudio.load(x1_path)
        wav2, _ = torchaudio.load(x2_path)

        wav1 = wav1.squeeze(0)
        wav2 = wav2.squeeze(0)

        length1 = wav1.shape[0]
        length2 = wav2.shape[0]

        return wav1, wav2, \
        torch.tensor(length1), torch.tensor(length2), \
        torch.tensor(int(y_label[0])),
    
    def collate_fn(self, data_sample):
        wavs1 = []
        wavs2 = []
        lengths1 = []
        lengths2 = []
        ylabels = []

        for samples in data_sample:
            wavs1.append(samples[0])
            wavs2.append(samples[1])
            lengths1.append(samples[2])
            lengths2.append(samples[3])
            ylabels.append(samples[4])

        all_wavs = []
        all_wavs.extend(wavs1)
        all_wavs.extend(wavs2)

        all_lengths = []
        all_lengths.extend(lengths1)
        all_lengths.extend(lengths2)

        return all_wavs, all_lengths, ylabels

