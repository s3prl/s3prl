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
import tqdm

# Voxceleb 2 Speaker verification
class SpeakerVerifi_train(Dataset):
    def __init__(self, file_path, meta_data, max_timestep=None):

        self.roots = file_path
        self.root_key = list(self.roots.keys())
        self.max_timestep = max_timestep

        # extract dev speaker and store in self.black_list_spealers
        with open(meta_data, "r") as f:
            self.black_list_speakers = f.read().splitlines()

        # calculate speakers and support to remove black list speaker (dev)
        self.all_speakers = \
            [f.path for key in self.root_key for f in os.scandir(self.roots[key]) if f.is_dir() and f.path.split("/")[-1] not in self.black_list_speakers]
        self.speaker_num = len(self.all_speakers)
        self.necessary_dict = self.processing()
        self.label_mapping_spk_id = {}
        # speaker id  map to speaker num
        self.build_label_mapping()

        print("search all wavs paths")
        start = time.time()
        self.dataset = []
        for speaker in tqdm.tqdm(self.all_speakers):
            wav_list=find_files(speaker)
            self.dataset.extend(wav_list)
        end = time.time() 
        print(f"search all wavs paths costs {end-start} seconds")

        self.label=self.build_label(self.dataset)

    def processing(self):
        
        speaker_num = len(self.all_speakers)
        return {"spk_paths":self.all_speakers,"total_spk_num":speaker_num,"pair_table":None}

    
    # file_path/id0001/asfsafs/xxx.wav
    def build_label_mapping(self):
        spk_count  = 0
        for speaker_id in self.all_speakers:
            self.label_mapping_spk_id[speaker_id.split("/")[-1]] = spk_count
            spk_count +=1
        
    
    def build_label(self,train_path_list):
        y = []
        for path in train_path_list:
            id_string = path.split("/")[-3]
            y.append(self.label_mapping_spk_id[id_string])

        return y
    
    def train(self):

        dataset = []
        for string in self.usage_list:
            pair = string.split()
            index = pair[0]
            x = os.path.join(self.root, pair[1])
            if int(index) == 1:
                dataset.append(x)
                
        return dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        
        wav, sr = torchaudio.load(self.dataset[idx])
        wav = wav.squeeze(0)
        length = wav.shape[0]

        if self.max_timestep !=None:
            if length > self.max_timestep:
                start = random.randint(0, int(length-self.max_timestep))
                wav = wav[start:start+self.max_timestep]
                length = self.max_timestep
  
        return wav, torch.tensor([length]), torch.tensor([self.label[idx]]).long()
        
    def collate_fn(self, samples):
        
        wavs, lengths, labels = [], [], []

        for wav,length,label in samples:
            wavs.append(wav)
            lengths.append(length)
            labels.append(label)
        return wavs, lengths, labels



class SpeakerVerifi_dev(Dataset):
    def __init__(self, file_path, meta_data, max_timestep=None):

        self.root = file_path
        self.meta_data = meta_data
        self.necessary_dict = self.processing()
        self.max_timestep = max_timestep
        self.dataset = self.necessary_dict['pair_table'] 
        
    def processing(self):
        pair_table = []
        with open(self.meta_data, "r") as f:
            usage_list = f.readlines()
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
        y_label, x1_path, x2_path = self.dataset[idx]
        wav1, _ = torchaudio.load(x1_path)
        wav2, _ = torchaudio.load(x2_path)

        wav1 = wav1.squeeze(0)
        wav2 = wav2.squeeze(0)

        length1 = wav1.shape[0]
        if self.max_timestep!=None:
            if length1 > self.max_timestep:
                length1 = self.max_timestep
                start = random.randint(0,length1 - self.max_timestep)
                wav1 = wav1[start:start+self.max_timestep]

        length2 = wav2.shape[0]

        if self.max_timestep!=None:
            if length2 > self.max_timestep:
                length2 = self.max_timestep
                start = random.randint(0,length2 - self.max_timestep)
                wav2 = wav1[start:start+self.max_timestep]


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



class SpeakerVerifi_test(Dataset):
    def __init__(self, file_path,meta_data=None):

        self.root = file_path
        self.meta_data = meta_data
        self.necessary_dict = self.processing()
        self.dataset = self.necessary_dict['pair_table'] 
        
    def processing(self):
        pair_table = []
        with open(self.meta_data, "r") as f:
            usage_list = f.readlines()
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
        y_label, x1_path, x2_path = self.dataset[idx]
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

