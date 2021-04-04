# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ dataset.py ]
#   Synopsis     [ the speech separation dataset ]
#   Author       [ Zili Huang ]
#   Copyright    [ Copyright(c), Johns Hopkins University ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import random

import numpy as np
import pandas as pd

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset

import librosa

class SeparationDataset(Dataset):
    def __init__(
        self,
        data_dir,
        rate=16000,
        src=['mix_clean'],
        tgt=['s1', 's2'],
        n_fft=512,
        hop_length=320,
        win_length=512,
        window='hann', 
        center=True,
    ):
        super(SeparationDataset, self).__init__()

        self.data_dir = data_dir
        self.rate = rate
        self.src = src
        self.tgt = tgt
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center
        self.n_srcs = len(self.tgt)

        assert len(self.src) == 1 and len(self.tgt) == 2

        # mix_clean (utterances only) mix_both (utterances + noise) mix_single (1 utterance + noise)
        cond_list = ["s1", "s2", "noise", "mix_clean", "mix_both", "mix_single"]
        # create the mapping from utterances to the audio paths
        # reco2path[uttname][cond]
        reco2path = {}
        for cond in src + tgt:
            assert cond in cond_list
            assert os.path.exists("{}/{}/wav.scp".format(self.data_dir, cond))
            with open("{}/{}/wav.scp".format(self.data_dir, cond), 'r') as fh:
                content = fh.readlines()
            for line in content:
                line = line.strip('\n')
                uttname, path = line.split()
                if uttname not in reco2path:
                    reco2path[uttname] = {}
                reco2path[uttname][cond] = path
        self.reco2path = reco2path

        self.recolist = list(self.reco2path.keys())
        self.recolist.sort()

    def __len__(self):
        return len(self.recolist)

    def __getitem__(self, i):
        reco = self.recolist[i]
        src_path = self.reco2path[reco][self.src[0]]
        src_samp, rate = librosa.load(src_path, sr=None)
        assert rate == self.rate
        src_feat = np.transpose(librosa.stft(src_samp, 
            n_fft=self.n_fft,
            hop_length = self.hop_length,
            win_length = self.win_length,
            window = self.window,
            center = self.center))

        tgt_samp_list, tgt_feat_list = [], []
        for j in range(self.n_srcs):
            tgt_path = self.reco2path[reco][self.tgt[j]]
            tgt_samp, rate = librosa.load(tgt_path, sr=None)
            assert rate == self.rate
            tgt_feat = np.transpose(librosa.stft(tgt_samp, 
                n_fft=self.n_fft,
                hop_length = self.hop_length,
                win_length = self.win_length,
                window = self.window,
                center = self.center))
            tgt_samp_list.append(tgt_samp)
            tgt_feat_list.append(tgt_feat)
        return reco, src_samp, src_feat, tgt_samp_list, tgt_feat_list

    def collate_fn(self, batch):
        sorted_batch = sorted(batch, key=lambda x: -x[1].shape[0])
        bs = len(sorted_batch)
        uttname_list = [sorted_batch[i][0] for i in range(bs)]

        # Store the magnitude, phase for the mixture in source_attr
        source_attr = {}
        mix_magnitude_list = [torch.from_numpy(np.abs(sorted_batch[i][2])) for i in range(bs)]
        mix_phase_list = [torch.from_numpy(np.angle(sorted_batch[i][2])) for i in range(bs)]
        mix_stft_list = [torch.from_numpy(sorted_batch[i][2]) for i in range(bs)]
        mix_magnitude = pad_sequence(mix_magnitude_list, batch_first=True)
        mix_phase = pad_sequence(mix_phase_list, batch_first=True)
        mix_stft = pad_sequence(mix_stft_list, batch_first=True)
        source_attr["magnitude"] = mix_magnitude
        source_attr["phase"] = mix_phase
        source_attr["stft"] = mix_stft

        target_attr = {}
        target_attr["magnitude"] = []
        target_attr["phase"] = []
        for j in range(self.n_srcs):
            tgt_magnitude_list = [torch.from_numpy(np.abs(sorted_batch[i][4][j])) for i in range(bs)]
            tgt_phase_list = [torch.from_numpy(np.angle(sorted_batch[i][4][j])) for i in range(bs)]
            tgt_magnitude = pad_sequence(tgt_magnitude_list, batch_first=True)
            tgt_phase = pad_sequence(tgt_phase_list, batch_first=True)
            target_attr["magnitude"].append(tgt_magnitude)
            target_attr["phase"].append(tgt_phase)

        wav_length = torch.from_numpy(np.array([len(sorted_batch[i][1]) for i in range(bs)]))
        source_wav_list = [torch.from_numpy(sorted_batch[i][1]) for i in range(bs)]
        source_wav = pad_sequence(source_wav_list, batch_first=True)
        target_wav_list = []
        for j in range(self.n_srcs):
            target_wav_list.append(pad_sequence([torch.from_numpy(sorted_batch[i][3][j]) for i in range(bs)], batch_first=True))

        feat_length = torch.from_numpy(np.array([stft.size(0) for stft in mix_stft_list]))
        return source_wav_list, uttname_list, source_attr, source_wav, target_attr, target_wav_list, feat_length, wav_length


if __name__ == '__main__':
    data_dir = "/export/c12/hzili1/tools/s3prl/downstream/separation/data/wav16k/min/train-100"
    rate = 16000
    src = ['mix_clean']
    tgt = ['s1', 's2']
    n_fft = 512
    hop_length = 320
    win_length = 512
    window = "hann"
    center = True
    bs = 8
    nworkers = 4

    dataset = SeparationDataset(
              data_dir=data_dir,
              rate=rate,
              src=src,
              tgt=tgt,
              n_fft=n_fft,
              hop_length=hop_length,
              win_length=win_length,
              window=window,
              center=center,
            ) 
    #for i, v in enumerate(dataset):
    #    uttname, src_wav, src_feat, target_wav_list, target_feat_list = v
    #    print(i)
    #    print("src_wav", src_wav.shape)
    #    print("src_feat", src_feat.shape)
    #    print("len(target_wav_list)", len(target_wav_list))
    #    print("len(target_feat_list)", len(target_feat_list))
    #    print("target_wav_list[0]", target_wav_list[0].shape)
    #    print("target_feat_list[0]", target_feat_list[0].shape)
    #    raise ValueError("debug")

    from torch.utils.data import DataLoader
    dataloader = DataLoader(
            dataset,
            batch_size=bs,
            shuffle=True,
            num_workers=nworkers,
            drop_last=False,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )
    for i, info in enumerate(dataloader):
        source_wav_list, uttname_list, source_attr, source_wav, target_attr, target_wav_list, feat_length, wav_length = info
        print("-" * 80)
        print("uttname_list", uttname_list)
        print("feat_length", feat_length)
        print("wav_length", wav_length)
        print("source_wav_list", len(source_wav_list))
        for j in range(len(source_wav_list)):
            print("source_wav_list[j]", source_wav_list[j].size())
        print("source_wav", source_wav.size())
        print("target_wav_list", target_wav_list[0].size())
        print("source_attr['magnitude']", source_attr['magnitude'].size())
        print("source_attr['phase']", source_attr['phase'].size())
        print("source_attr['stft']", source_attr['stft'].size())
        print("target_attr['magnitude']", target_attr['magnitude'][0].size())
        print("target_attr['phase']", target_attr['phase'][0].size())
        break
