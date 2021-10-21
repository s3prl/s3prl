# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ dataset.py ]
#   Synopsis     [ LibriMix speech enhancement dataset ]
#   Author       [ Zili Huang ]
#   Copyright    [ Copyright(c), Johns Hopkins University ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import random

import numpy as np

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset

import librosa

SAMPLE_RATE = 16000

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
        """
        Args:
            data_dir (str):
                prepared data directory

            rate (int):
                audio sample rate

            src and tgt (list(str)):
                the input and desired output.
                LibriMix offeres different options for the users. For
                clean source separation, src=['mix_clean'] and tgt=['s1', 's2'].
                Please see https://github.com/JorisCos/LibriMix for details

            n_fft (int):
                length of the windowed signal after padding with zeros.

            hop_length (int):
                number of audio samples between adjacent STFT columns.

            win_length (int):
                length of window for each frame

            window (str):
                type of window function, only support Hann window now

            center (bool):
                whether to pad input on both sides so that the
                t-th frame is centered at time t * hop_length

            The STFT related parameters are the same as librosa.
        """

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

        assert len(self.src) == 1 and len(self.tgt) == 1

        # mix_clean (utterances only) mix_both (utterances + noise) mix_single (1 utterance + noise)
        cond_list = ["s1", "s2", "noise", "mix_clean", "mix_both", "mix_single", "noisy", "clean"]

        # create the mapping from utterances to the audio paths
        # reco2path[utt][cond] is the path for utterance utt with condition cond
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
        src_samp, rate = librosa.load(src_path, sr=SAMPLE_RATE)
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
            tgt_samp, rate = librosa.load(tgt_path, sr=SAMPLE_RATE)
            assert rate == self.rate
            tgt_feat = np.transpose(librosa.stft(tgt_samp, 
                n_fft=self.n_fft,
                hop_length = self.hop_length,
                win_length = self.win_length,
                window = self.window,
                center = self.center))
            tgt_samp_list.append(tgt_samp)
            tgt_feat_list.append(tgt_feat)
        """
        reco (str):
            name of the utterance

        src_sample (ndarray):
            audio samples for the source [T, ]

        src_feat (ndarray):
            the STFT feature map for the source with shape [T1, D]

        tgt_samp_list (list(ndarray)):
            list of audio samples for the targets

        tgt_feat_list (list(ndarray)):
            list of STFT feature map for the targets
        """
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
        """
        source_wav_list (list(tensor)):
            list of audio samples for the source

        uttname_list (list(str)):
            list of utterance names

        source_attr (dict):
            dictionary containing magnitude and phase information for the sources

        source_wav (tensor):
            padded version of source_wav_list, with size [bs, max_T]

        target_attr (dict):
            dictionary containing magnitude and phase information for the targets

        feat_length (tensor):
            length of the STFT feature for each utterance

        wav_length (tensor):
            number of samples in each utterance
        """
        return source_wav_list, uttname_list, source_attr, source_wav, target_attr, target_wav_list, feat_length, wav_length
