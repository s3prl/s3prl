# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ dataset.py ]
#   Synopsis     [ The VCTK + VCC2020 dataset ]
#   Author       [ Wen-Chin Huang (https://github.com/unilight) ]
#   Copyright    [ Copyright(c), Toda Lab, Nagoya University, Japan ]
"""*********************************************************************************************"""


import os
import random
import yaml

import librosa
import numpy as np
from tqdm import tqdm

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset

from resemblyzer import preprocess_wav, VoiceEncoder
from .utils import logmelspectrogram
from .utils import read_hdf5, write_hdf5

SRCSPKS = ["SEF1", "SEF2", "SEM1", "SEM2"]
TRGSPKS_TASK1 = ["TEF1", "TEF2", "TEM1", "TEM2"]
FS = 16000 # Always resample to 16kHz

def generate_eval_pairs(file_list, train_file_list, eval_data_root, num_samples):
    X = []
    for trgspk in TRGSPKS_TASK1:
        # filter out those exist
        spk_file_list = []
        for number in train_file_list:
            wav_path = os.path.join(eval_data_root, trgspk, number + ".wav")
            if os.path.isfile(wav_path):
                spk_file_list.append(wav_path)
        # generate pairs
        for srcspk in SRCSPKS:
            for number in file_list:
                random.shuffle(spk_file_list)
                pair = [os.path.join(eval_data_root, srcspk, number + ".wav")]
                pair.extend(spk_file_list[:num_samples])
                X.append(pair)
    return X

class VCTK_VCC2020Dataset(Dataset):
    def __init__(self, split, 
                 trdev_data_root, eval_data_root, spk_embs_root, 
                 lists_root, eval_lists_root,
                 fbank_config, spk_emb_source, num_ref_samples,
                 train_dev_seed=1337, **kwargs):
        super(VCTK_VCC2020Dataset, self).__init__()
        self.split = split
        self.fbank_config = fbank_config
        self.spk_emb_source = spk_emb_source
        self.spk_embs_root = spk_embs_root
        os.makedirs(spk_embs_root, exist_ok=True)

        X = []
        if split == 'train' or split == 'dev':
            file_list = open(os.path.join(lists_root, split + '_list.txt')).read().splitlines()
            for fname in file_list:
                spk, number = fname.split("_")
                wav_path = os.path.join(trdev_data_root, spk, fname + ".wav")
                X.append(wav_path)
            random.seed(train_dev_seed)
            random.shuffle(X)
        elif split == 'test':
            for num_samples in num_ref_samples:
                # goal: save converted samples with diff num of ref samples to different folders?
                eval_pair_list_file = os.path.join(lists_root, "eval_{}sample_list.txt".format(num_samples))
                if os.path.isfile(eval_pair_list_file):
                    print("[Dataset] eval pair list file exists: {}".format(eval_pair_list_file))
                    with open(eval_pair_list_file, "r") as f:
                        lines = f.read().splitlines()
                    X += [line.split(",") for line in lines]
                else:
                    print("[Dataset] eval pair list file does not exist: {}".format(eval_pair_list_file))
                    # generate eval pairs
                    file_list = open(os.path.join(eval_lists_root, 'eval_list.txt')).read().splitlines()
                    train_file_list = open(os.path.join(eval_lists_root, 'E_train_list.txt')).read().splitlines()
                    eval_pairs = generate_eval_pairs(file_list, train_file_list, eval_data_root, num_samples)
                    # write in file
                    with open(eval_pair_list_file, "w") as f:
                        for line in eval_pairs:
                            f.write(",".join(line)+"\n")
                    X += eval_pairs
        else:
            raise ValueError('Invalid \'split\' argument for dataset: VCTK_VCC2020Dataset!')
        print('[Dataset] - number of data for ' + split + ': ' + str(len(X)))
        self.X = X

        if spk_emb_source == "external":
            # extract spk embs beforehand
            print("[Dataset] Extracting speaker emebddings")
            self.extract_spk_embs()
        else:
            NotImplementedError

    def extract_spk_embs(self):
        # load speaker encoder
        spk_encoder = VoiceEncoder()

        if self.split == "train" or self.split == "dev":
            spk_emb_paths = [os.path.join(self.spk_embs_root, os.path.basename(wav_path).replace(".wav", ".h5")) for wav_path in self.X]
            self.X = list(zip(self.X, spk_emb_paths))
            for wav_path, spk_emb_path in tqdm(self.X, dynamic_ncols=True, desc="Extracting speaker embedding"):
                if not os.path.isfile(spk_emb_path):
                    # extract spk emb
                    wav = preprocess_wav(wav_path)
                    embedding = spk_encoder.embed_utterance(wav)
                    # save spk emb
                    write_hdf5(spk_emb_path, "spk_emb", embedding.astype(np.float32))
        elif self.split == "test":
            new_X = []
            for wav_paths in self.X:
                source_wav_path = wav_paths[0]
                new_tuple = [source_wav_path]
                for wav_path in wav_paths[1:]:
                    spk, number = wav_path.split(os.sep)[-2:]
                    spk_emb_path = os.path.join(self.spk_embs_root, spk + "_" + number.replace(".wav", ".h5"))
                    new_tuple.append(spk_emb_path)
                    if not os.path.isfile(spk_emb_path):
                        # extract spk emb
                        wav = preprocess_wav(wav_path)
                        embedding = spk_encoder.embed_utterance(wav)
                        # save spk emb
                        write_hdf5(spk_emb_path, "spk_emb", embedding.astype(np.float32))
                new_X.append(new_tuple)
            self.X = new_X

    def _load_wav(self, wav_path, fs):
        # use librosa to resample. librosa gives range [-1, 1]
        wav, sr = librosa.load(wav_path, sr=fs)
        return wav, sr

    def __len__(self):
        return len(self.X)

    def get_all_lmspcs(self):
        lmspcs = []
        for xs in tqdm(self.X, dynamic_ncols=True, desc="Extracting target acoustic features"):
            input_wav_path = xs[0]
            input_wav_original, fs_original = self._load_wav(input_wav_path, fs=None)
            lmspc = logmelspectrogram(
                x=input_wav_original,
                fs=fs_original,
                n_mels=self.fbank_config["n_mels"],
                n_fft=self.fbank_config["n_fft"],
                n_shift=self.fbank_config["n_shift"],
                win_length=self.fbank_config["win_length"],
                window=self.fbank_config["window"],
                fmin=self.fbank_config["fmin"],
                fmax=self.fbank_config["fmax"],
            )
            lmspcs.append(lmspc)
        return lmspcs
        

    def __getitem__(self, index):
        input_wav_path = self.X[index][0]
        spk_emb_paths = self.X[index][1:]
        ref_spk_name = os.path.basename(spk_emb_paths[0]).split("_")[0]

        input_wav_original, _ = self._load_wav(input_wav_path, fs=self.fbank_config["fs"])
        input_wav_resample, fs_resample = self._load_wav(input_wav_path, fs=FS)

        lmspc = logmelspectrogram(
            x=input_wav_original,
            fs=self.fbank_config["fs"],
            n_mels=self.fbank_config["n_mels"],
            n_fft=self.fbank_config["n_fft"],
            n_shift=self.fbank_config["n_shift"],
            win_length=self.fbank_config["win_length"],
            window=self.fbank_config["window"],
            fmin=self.fbank_config["fmin"],
            fmax=self.fbank_config["fmax"],
        )

        # get speaker embeddings
        if self.spk_emb_source == "external":
            ref_spk_embs = [read_hdf5(spk_emb_path, "spk_emb") for spk_emb_path in spk_emb_paths]
            ref_spk_embs = np.stack(ref_spk_embs, axis=0)
            ref_spk_emb = np.mean(ref_spk_embs, axis=0)
        else:
            ref_spk_emb = None

        # change input wav path name
        if self.split == "test":
            input_wav_name = input_wav_path.replace(".wav", "")
            input_wav_path = input_wav_name + "_{}samples.wav".format(len(spk_emb_paths))

        return input_wav_resample, input_wav_original, lmspc, ref_spk_emb, input_wav_path, ref_spk_name
    
    def collate_fn(self, batch):
        sorted_batch = sorted(batch, key=lambda x: -x[1].shape[0])
        bs = len(sorted_batch) # batch_size
        wavs = [torch.from_numpy(sorted_batch[i][0]) for i in range(bs)]
        wavs_2 = [torch.from_numpy(sorted_batch[i][1]) for i in range(bs)] # This is used for obj eval
        acoustic_features = [torch.from_numpy(sorted_batch[i][2]) for i in range(bs)]
        acoustic_features_padded = pad_sequence(acoustic_features, batch_first=True)
        acoustic_feature_lengths = torch.from_numpy(np.array([acoustic_feature.size(0) for acoustic_feature in acoustic_features]))
        ref_spk_embs = torch.from_numpy(np.array([sorted_batch[i][3] for i in range(bs)]))
        wav_paths = [sorted_batch[i][4] for i in range(bs)]
        ref_spk_names = [sorted_batch[i][5] for i in range(bs)]
        
        return wavs, wavs_2, acoustic_features, acoustic_features_padded, acoustic_feature_lengths, wav_paths, ref_spk_embs, ref_spk_names, None


class CustomDataset(Dataset):
    def __init__(self,
                 eval_pair_list_file,
                 spk_emb_source,
                 **kwargs):
        super(CustomDataset, self).__init__()
        self.spk_emb_source = spk_emb_source

        if os.path.isfile(eval_pair_list_file):
            print("[Dataset] Reading custom eval pair list file: {}".format(eval_pair_list_file))
            with open(eval_pair_list_file, "r") as f:
                infos = yaml.load(f, Loader=yaml.FullLoader)
            X = [{"wav_name": k, **v} for k, v in infos.items()]
        else:
            raise ValueError("[Dataset] eval pair list file does not exist: {}".format(eval_pair_list_file))
        print('[Dataset] - number of data for custom test: ' + str(len(X)))
        self.X = X

        if spk_emb_source == "external":
            # extract spk embs beforehand
            print("[Dataset] Extracting speaker emebddings")
            self.extract_spk_embs()
        else:
            NotImplementedError

    def extract_spk_embs(self):
        # load speaker encoder
        spk_encoder = VoiceEncoder()

        new_X = []
        for item in self.X:
            new_item = item
            new_item["ref_spk_embs"] = []
            for wav_path in new_item["ref"]:
                # extract spk emb
                wav = preprocess_wav(wav_path)
                embedding = spk_encoder.embed_utterance(wav)
                new_item["ref_spk_embs"].append(embedding)
            new_X.append(new_item)
        self.X = new_X

    def _load_wav(self, wav_path, fs):
        # use librosa to resample. librosa gives range [-1, 1]
        wav, sr = librosa.load(wav_path, sr=fs)
        return wav, sr

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        wav_name = self.X[index]["wav_name"]
        input_wav_path = self.X[index]["src"]
        ref_spk_embs = self.X[index]["ref_spk_embs"]
        ref_spk_name = self.X[index]["ref_spk_name"]

        input_wav_original, _ = self._load_wav(input_wav_path, fs=None)
        input_wav_resample, fs_resample = self._load_wav(input_wav_path, fs=FS)

        # get speaker embeddings
        if self.spk_emb_source == "external":
            # ref_spk_embs = [read_hdf5(spk_emb_path, "spk_emb") for spk_emb_path in spk_emb_paths]
            ref_spk_embs = np.stack(ref_spk_embs, axis=0)
            ref_spk_emb = np.mean(ref_spk_embs, axis=0)
        else:
            ref_spk_emb = None

        return input_wav_resample, input_wav_original, ref_spk_emb, input_wav_path, ref_spk_name, wav_name

    def collate_fn(self, batch):
        sorted_batch = sorted(batch, key=lambda x: -x[1].shape[0])
        bs = len(sorted_batch) # batch_size
        wavs = [torch.from_numpy(sorted_batch[i][0]) for i in range(bs)]
        wavs_2 = [torch.from_numpy(sorted_batch[i][1]) for i in range(bs)] # This is used for obj eval
        ref_spk_embs = torch.from_numpy(np.array([sorted_batch[i][2] for i in range(bs)]))
        wav_paths = [sorted_batch[i][3] for i in range(bs)]
        ref_spk_names = [sorted_batch[i][4] for i in range(bs)]
        save_wav_names = [sorted_batch[i][5] for i in range(bs)]
        
        return wavs, wavs_2, None, None, None, wav_paths, ref_spk_embs, ref_spk_names, save_wav_names