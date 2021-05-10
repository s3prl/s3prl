import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np 
from librosa.util import find_files
from torchaudio import load
from torch import nn
from pathlib import Path
import os
import random
import torchaudio
import sys
import time
import tqdm
import pickle
from torchaudio.sox_effects import apply_effects_file

EFFECTS = [
["channels", "1"],
["rate", "16000"],
["gain", "-3.0"],
["silence", "1", "0.1", "0.1%", "-1", "0.1", "0.1%"],
]

# Voxceleb 2 Speaker verification
class SpeakerVerifi_train(Dataset):
    def __init__(self, vad_config, file_path, meta_data, max_timestep=None):

        self.roots = file_path
        self.root_key = list(self.roots.keys())
        self.max_timestep = max_timestep
        self.vad_c = vad_config 
        self.dataset = []
        self.all_speakers = []

        for key in self.root_key:
            
            cache_path = f"./downstream/voxceleb2_amsoftmax_segment_eval/cache_wav_paths/cache_{key}.p"
            p = Path(self.roots[key])
            # loca cache_path if file exists
            if os.path.isfile(cache_path):

                # cache dict = 
                #{"speaker_id1":["wav_a_path1","wav_a_path2",...],"speaker_id2":["wav_b_path1", "wav_b_path2", ....],...}
                cache_wavs_dict = pickle.load(open(cache_path,"rb"))
                self.all_speakers.extend(list(cache_wavs_dict.keys()))
                for speaker_id in list(cache_wavs_dict.keys()):
                    for wavs in cache_wavs_dict[speaker_id]:
                        self.dataset.append(str(p / speaker_id / wavs))

            else:

                speaker_wav_dict = {}
                # calculate speakers and support to remove black list speaker (dev)
                speaker_dirs = [f.path.split("/")[-1] for f in os.scandir(self.roots[key]) if f.is_dir()]
                self.all_speakers.extend(speaker_dirs)
                    
                print("search all wavs paths")
                start = time.time()

                for speaker in tqdm.tqdm(speaker_dirs):
                    speaker_dir =  p / speaker
                    wav_list=find_files(speaker_dir)
                    speaker_wav_dict[speaker] = []

                    for wav in wav_list:

                        wav, _ = apply_effects_file(str(speaker_dir/wav), EFFECTS)
                        wav = wav.squeeze(0)
                        length = wav.shape[0]

                        if length > self.vad_c['min_sec']: 
                            self.dataset.append(str(speaker_dir/wav))
                            speaker_wav_dict[speaker].append("/".join(wav.split("/")[-2:]))
                end = time.time() 
                print(f"search all wavs paths costs {end-start} seconds")
                print(f"save wav paths to {cache_path}! so we can directly load all_path in next time!")
                pickle.dump(speaker_wav_dict, open(cache_path,"wb"))   

        self.speaker_num = len(self.all_speakers)
        self.necessary_dict = self.processing()
        self.label_mapping_spk_id = {}
        # speaker id  map to speaker num
        self.build_label_mapping()

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

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        wav, _ = torchaudio.load(self.dataset[idx])
        # wav, _ = apply_effects_file(self.dataset[idx], EFFECTS)
        wav = wav.squeeze(0)
        length = wav.shape[0]
        
        if self.max_timestep !=None:
            if length > self.max_timestep:
                start = random.randint(0, int(length-self.max_timestep))
                wav = wav[start:start+self.max_timestep]
                length = self.max_timestep
  
        return wav, torch.tensor([self.label[idx]]).long()
        
    def collate_fn(self, samples):
        
        wavs, labels = [], []
        None_list1= []
        None_list2= []        
        None_list3= []

        for wav,label in samples:
            wavs.append(wav)
            labels.append(label)
            None_list1.append(None)
            None_list2.append(None)
            None_list3.append(None)

        return wavs, labels, None_list1, None_list2, None_list3



class SpeakerVerifi_dev(Dataset):
    def __init__(self, vad_config, segment_config, file_path, meta_data):

        self.root = file_path
        self.meta_data = meta_data
        self.segment_config = segment_config
        self.vad_c = vad_config
        self.pair_dict = self.preprocessing()

        cache_path = f"./downstream/voxceleb2_amsoftmax_segment_eval/cache_wav_paths/cache_dev_segment.p"
        # loca cache_path if file exists
        if os.path.isfile(cache_path):
            self.dataset=pickle.load(open(cache_path,"rb"))
        else:
            self.dataset = self.segment_processing()
            pickle.dump(self.dataset, open(cache_path,"wb"))

    
    def segment_processing(self):
        wav_list = self.pair_dict['wav_table']
        utterance_id = 0
        segment_list = []
        print("processing test set to segments")
        for wav_info in tqdm.tqdm(wav_list):
            label_info = wav_info[0]
            pair_info = wav_info[1]

            wav, _ = apply_effects_file(wav_info[2], EFFECTS)
            wav = wav.squeeze(0)

            index_end = len(wav) -self.segment_config["window"]
            segment_num = index_end // self.segment_config['stride']

            if index_end < 0:
                segment_list.append([int(label_info), pair_info, str(utterance_id), segment_num, 0, len(wav), wav_info[2]])
            else:
                for index in range(0, index_end, self.segment_config['stride']):
                    segment_list.append([int(label_info), pair_info, str(utterance_id), segment_num, index, index+self.segment_config['window'], wav_info[2]])

            utterance_id += 1
            
        return segment_list

        
    def preprocessing(self):
        wav_table = []
        pair_id = 0 
        with open(self.meta_data, "r") as f:
            usage_list = f.readlines()
        for pair in usage_list:
            list_pair = pair.split()
            pair_1= os.path.join(self.root, list_pair[1])
            pair_2= os.path.join(self.root, list_pair[2])
            
            wav1 = (list_pair[0], str(pair_id), pair_1)
            wav2 = (list_pair[0], str(pair_id), pair_2)
            
            wav_table.append(wav1)
            wav_table.append(wav2)
            
            pair_id +=1

        return {"wav_table":wav_table}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        label_info, pair_id, utter_id, seg_info, start, end, path = self.dataset[idx]
        wav, _ = torchaudio.load(path)
        # wav, _ = apply_effects_file(path, EFFECTS)
        wav = wav.squeeze(0)
        seg_tensor = wav[start:end]

        return label_info, pair_id, utter_id, seg_info, seg_tensor

    
    def collate_fn(self, data_sample):
        label_list = []
        pair_list = []
        utterid_list = []
        seg_num_list = []
        seg_tensor_list = []

        for samples in data_sample:
            label_list.append(samples[0])
            pair_list.append(samples[1])
            utterid_list.append(samples[2])
            seg_num_list.append(samples[3])
            seg_tensor_list.append(samples[4])

        return seg_tensor_list, label_list, pair_list, utterid_list, seg_num_list

class SpeakerVerifi_test(Dataset):
    def __init__(self, vad_config, segment_config, file_path, meta_data):
    
        self.root = file_path
        self.meta_data = meta_data
        self.segment_config = segment_config
        self.vad_c = vad_config
        self.pair_dict = self.preprocessing()

        cache_path = f"./downstream/voxceleb2_amsoftmax_segment_eval/cache_wav_paths/cache_test_segment.p"
        # loca cache_path if file exists
        if os.path.isfile(cache_path):
            self.dataset=pickle.load(open(cache_path,"rb"))
        else:
            self.dataset = self.segment_processing()
            pickle.dump(self.dataset, open(cache_path,"wb"))
    
    def segment_processing(self):
        wav_list = self.pair_dict['wav_table']
        utterance_id = 0
        segment_list = []
        print("processing test set to segments")
        for wav_info in tqdm.tqdm(wav_list):
            label_info = wav_info[0]
            pair_info = wav_info[1]
            wav, _ = torchaudio.load(wav_info[2])
            # wav, _ = apply_effects_file(wav_info[2], EFFECTS)
            wav = wav.squeeze(0)

            index_end = len(wav) -self.segment_config["window"]
            segment_num = index_end // self.segment_config['stride']

            if index_end < 0:
                segment_list.append([int(label_info), pair_info, str(utterance_id), segment_num, 0, len(wav), wav_info[2]])
            else:
                for index in range(0, index_end, self.segment_config['stride']):
                    segment_list.append([int(label_info), pair_info, str(utterance_id), segment_num, index, index+self.segment_config['window'], wav_info[2]])

            utterance_id += 1
            
        return segment_list

        
    def preprocessing(self):
        wav_table = []
        pair_id = 0 
        with open(self.meta_data, "r") as f:
            usage_list = f.readlines()
        for pair in usage_list:
            list_pair = pair.split()
            pair_1= os.path.join(self.root, list_pair[1])
            pair_2= os.path.join(self.root, list_pair[2])
            
            wav1 = (list_pair[0], str(pair_id), pair_1)
            wav2 = (list_pair[0], str(pair_id), pair_2)
            
            wav_table.append(wav1)
            wav_table.append(wav2)
            
            pair_id +=1

        return {"wav_table":wav_table}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        label_info, pair_id, utter_id, seg_info, start, end, path = self.dataset[idx]
        
        wav, _ = torchaudio.load(path)
        # wav, _ = apply_effects_file(path, EFFECTS)
        wav = wav.squeeze(0)
        seg_tensor = wav[start:end]

        return label_info, pair_id, utter_id, seg_info, seg_tensor

    
    def collate_fn(self, data_sample):
        label_list = []
        pair_list = []
        utterid_list = []
        seg_num_list = []
        seg_tensor_list = []

        for samples in data_sample:
            label_list.append(samples[0])
            pair_list.append(samples[1])
            utterid_list.append(samples[2])
            seg_num_list.append(samples[3])
            seg_tensor_list.append(samples[4])

        return seg_tensor_list, label_list, pair_list, utterid_list, seg_num_list

