import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np 
from librosa.util import find_files
from torchaudio import load
from torch import nn
import os 
import IPython 
import pdb
import random
import torchaudio
import sys
import time

# Voxceleb 1 Speaker Identification
class SpeakerClassificationDataset(Dataset):
    def __init__(self, file_path, max_timestep=500, n_mels=80, n_linear=1025, meta_data=None, is_dev=False, is_eval=False, proportion=None):


        self.root = file_path
        self.proportion = proportion
        self.speaker_num = 1251
        self.is_eval = is_eval
        self.meta_data =meta_data
        self.is_dev = is_dev

        # Read file
        if not self.is_eval:        
            usage_list = open(self.meta_data, "r").readlines()
            self.test_dataset = []
            self.test_label = []
            self.training_dataset = []
            self.dev_dataset = []
            self.dev_label = []
            self.train_label = []

            for string in usage_list:
                pair = string.split()
                index = pair[0]
                x = os.path.join(self.root, pair[1])
                if int(index) == 1:
                    self.training_dataset.append(x)
                elif int(index) == 2:
                    self.dev_dataset.append(x)
                elif int(index) == 3:
                    self.test_dataset.append(x)
            
            self.training_label = self.build_label(self.training_dataset)
            self.dev_label = self.build_label(self.dev_dataset)
            self.test_label = self.build_label(self.test_dataset)
            # Crop seqs that are too long
            self.max_timestep = max_timestep

        else:
            usage_list = open(self.meta_data, "r").readlines()
            self.test_dataset = []
            self.test_label = []
            for string in usage_list:
                pair = string.split()
                index = pair[0]
                if int(index) == 3:
                    x = os.path.join(self.root, pair[1])
                    self.test_dataset.append(x)
            self.test_label = self.build_label(self.test_dataset)

    # file_path/id0001/asfsafs/xxx.wav
    def build_label(self, train_path_list):
        y = []
        for path in train_path_list:
            id_string = path.split("/")[-3]
            y.append(int(id_string[2:]) - 10001)
        return y
            

    def __len__(self):
        if self.is_eval:
            return len(self.test_dataset)
        elif self.is_dev:
            return len(self.dev_dataset)
        else:
            return len(self.training_dataset)
    
    def __getitem__(self, idx):
        if self.is_dev:
            wav, sr = torchaudio.load(self.dev_dataset[idx])
            wav = wav.squeeze(0)
            length = wav.shape[0]
            return wav, torch.tensor([length]), torch.tensor([self.dev_label[idx]])
        elif self.is_eval:
            wav, sr = torchaudio.load(self.test_dataset[idx])
            wav = wav.squeeze(0)
            length = wav.shape[0]
            return wav, torch.tensor([length]), torch.tensor([self.test_label[idx]])
        else:
            wav, sr =torchaudio.load(self.training_dataset[idx])
            wav = wav.squeeze(0)
            length = wav.shape[0]
            pad_length = None 

            if length < self.max_timestep:
                x = wav                
            else:
                x = wav[:self.max_timestep]
                length = self.max_timestep
            return x, torch.tensor([length]), torch.tensor([self.training_label[idx]]).long()
    
    def collate_fn(self, samples):
        wavs, lengths, labels = [], [], []
        for wav,length,label in samples:
            wavs.append(wav)
            lengths.append(length)
            labels.append(label)
        return wavs, lengths, labels
