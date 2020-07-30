# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ dataloader.py ]
#   Synopsis     [ Datasets for transformer pre-training and downstream task supervised training ]
#   Author       [ Andy T. Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import torch
import pickle
import random
from functools import partial
import numpy as np
import pandas as pd
from tqdm import tqdm
from librosa.util import find_files
import torchaudio
from torchaudio.transforms import Spectrogram, MelScale, MFCC
from torchaudio.functional import magphase, compute_deltas
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformer.mam import process_train_MAM_data, process_test_MAM_data


############
# CONSTANT #
############
HALF_BATCHSIZE_TIME = 1000
SPEAKER_THRESHOLD = 0


#####################################################
# OnlineDataset: Only support pretraining currently #
#####################################################
class OnlinePreprocessor(torch.nn.Module):
    def __init__(self, sample_rate=16000, win_ms=25, hop_ms=10, n_freq=201, n_mels=40, n_mfcc=13, feat_list=None, **kwargs):
        super(OnlinePreprocessor, self).__init__()
        win = round(win_ms * sample_rate / 1000)
        hop = round(hop_ms * sample_rate / 1000)
        n_fft = (n_freq - 1) * 2
        self._win_args = {'n_fft': n_fft, 'hop_length': hop, 'win_length': win}
        self.register_buffer('_window', torch.hann_window(win))
        
        self._stft_args = {'center': True, 'pad_mode': 'reflect', 'normalized': False, 'onesided': True}
        # stft_args: same default values as torchaudio.transforms.Spectrogram & librosa.core.spectrum._spectrogram
        self._stft = partial(torch.stft, **self._win_args, **self._stft_args)
        self._magphase = partial(torchaudio.functional.magphase, power=2)
        self._melscale = MelScale(sample_rate=sample_rate, n_mels=n_mels)
        self._mfcc_trans = MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc, log_mels=True, melkwargs=self._win_args)
        self._istft = partial(torchaudio.functional.istft, **self._win_args, **self._stft_args)
        
        self.feat_list = feat_list
        self.register_buffer('_pseudo_wav', torch.randn(2, 2, sample_rate))  # batch_size=2, channel_size=2
    
    def _check_list(self, feat_list):
        if feat_list is None:
            feat_list = self.feat_list
        assert type(feat_list) is list
        return feat_list

    def _transpose_list(self, feats):
        return [feat.transpose(-1, -2).contiguous() if type(feat) is torch.Tensor else feat for feat in feats]

    def forward(self, wavs=None, feat_list=None):
        # wavs: (*, channel_size, max_len)
        # feat_list, mam_list: [{feat_type: 'mfcc', channel: 0, log: False, delta: 2, cmvn: 'True'}, ...]
        feat_list = self._check_list(feat_list)
        if wavs is None:
            max_channel_id = max([int(args['channel']) if 'channel' in args else 0 for args in feat_list])
            wavs = self._pseudo_wav.expand(-1, max_channel_id + 1, -1)
        assert wavs.dim() >= 3
        
        shape = wavs.size()
        complx = self._stft(wavs.reshape(-1, shape[-1]), window=self._window)
        complx = complx.reshape(shape[:-1] + complx.shape[-3:])
        # complx: (*, channel_size, feat_dim, max_len, 2)
        linear, phase = self._magphase(complx)
        mel = self._melscale(linear)
        mfcc = self._mfcc_trans(wavs)
        complx = complx.transpose(-1, -2).reshape(*mfcc.shape[:2], -1, mfcc.size(-1))
        # complx, linear, phase, mel, mfcc: (*, channel_size, feat_dim, max_len)

        def select_feat(variables, feat_type, channel=0, log=False, delta=0, cmvn=False):
            raw_feat = variables[feat_type].select(dim=-3, index=channel)
            # apply log scale
            if bool(log):
                raw_feat = (raw_feat + 1e-10).log()   
            feats = [raw_feat.contiguous()]
            # apply delta for features
            for _ in range(int(delta)):
                feats.append(compute_deltas(feats[-1]))
            feats = torch.cat(feats, dim=-2)
            # apply cmvn
            if bool(cmvn):
                feats = (feats - feats.mean(dim=-1, keepdim=True)) / (feats.std(dim=-1, keepdim=True) + 1e-10)
            return feats
            # return: (*, feat_dim, max_len)
        
        local_variables = locals()
        return self._transpose_list([select_feat(local_variables, **args) for args in feat_list])
        # return: [(*, max_len, feat_dim), ...]

    def istft(self, complxs=None, linears=None, phases=None, linear_power=2):
        assert complxs is not None or (linears is not None and phases is not None)
        # complxs: (*, n_freq, max_feat_len, 2) or (*, max_feat_len, n_freq * 2)
        # linears, phases: (*, max_feat_len, n_freq)

        if complxs is None:
            linears, phases = self._transpose_list([linears, phases])
            complxs = linears.pow(1/linear_power).unsqueeze(-1) * torch.stack([phases.cos(), phases.sin()], dim=-1)
        if complxs.size(-1) != 2:
            # treat complxs as: (*, max_feat_len, n_freq * 2)
            shape = complxs.size()
            complxs = complxs.view(*shape[:-1], -1, 2).transpose(-2, -3).contiguous()
        # complxs: (*, n_freq, max_feat_len, 2)
        
        return self._istft(complxs, window=self._window)
        # return: (*, max_wav_len)

    def test_istft(self, wavs=None, epsilon=1e-6):
        # wavs: (*, channel_size, max_wav_len)
        if wavs is None:
            wavs = self._pseudo_wav

        channel1, channel2 = 0, 1
        feat_list = [
            {'feat_type': 'complx', 'channel': channel1},
            {'feat_type': 'linear', 'channel': channel2},
            {'feat_type': 'phase', 'channel': channel2}
        ]
        complxs, linears, phases = self.forward(wavs, feat_list)
        assert torch.allclose(wavs.select(dim=-2, index=channel1), self.istft(complxs=complxs), atol=epsilon)
        assert torch.allclose(wavs.select(dim=-2, index=channel2), self.istft(linears=linears, phases=phases), atol=epsilon)
        print('[Test passed] stft -> istft')


def get_online_Dataloader(args, config, is_train=True):
    # create waveform dataset
    dataset = OnlineDataset(**config['online'])
    
    # create dataloader for extracting features
    def collate_fn(samples):
        # samples: [(seq_len, channel), ...]
        samples = pad_sequence(samples, batch_first=True)
        # samples: (batch_size, max_len, channel)
        return samples.transpose(-1, -2).contiguous()
        # return: (batch_size, channel, max_len)
        
    dataloader = DataLoader(dataset, batch_size=config['dataloader']['batch_size'],
                            shuffle=is_train, num_workers=config['dataloader']['n_jobs'],
                            pin_memory=True, collate_fn=collate_fn)
    return dataloader


class OnlineDataset(Dataset):
    def __init__(self, roots, sample_rate, max_time, target_level, noise_proportion, snrs, **kwargs):
        self.sample_rate = sample_rate
        self.max_time = max_time
        self.target_level = target_level
        self.noise_proportion = noise_proportion
        self.snrs = snrs
        self.filepths = []
        for root in roots:
            self.filepths += find_files(root)
        assert len(self.filepths) > 0, 'No audio file detected'
        self.noise_sampler = torch.distributions.Normal(0, 1)
    
    def _normalize(self, audio):
        '''Normalize the signal to the target level'''
        rms = audio.pow(2).mean().pow(0.5)
        scalar = (10 ** (self.target_level / 20)) / (rms + 1e-10)
        audio = audio * scalar
        return audio

    def __getitem__(self, idx):
        wav, sr = torchaudio.load(self.filepths[idx])
        assert sr == self.sample_rate, f'Sample rate mismatch: real {sr}, config {self.sample_rate}'
        wav = wav.view(-1)
        maxpoints = int(sr / 1000) * self.max_time
        if len(wav) > maxpoints:
            start = random.randint(0, len(wav) - maxpoints)
            wav = wav[start:start + maxpoints]
        wav = self._normalize(wav)

        # build input
        dice = random.random()
        if dice < self.noise_proportion:
            noise = self.noise_sampler.sample(wav.shape)
            snr = float(self.snrs[random.randint(0, len(self.snrs) - 1)])
            snr_exp = 10.0 ** (snr / 10.0)
            wav_power = wav.pow(2).sum()
            noise_power = noise.pow(2).sum()
            scalar = (wav_power / (snr_exp * noise_power)).pow(0.5)
            wav_inp = wav + scalar * noise
        else:
            wav_inp = wav

        # build target
        wav_tar = wav

        return torch.stack([wav_inp, wav_tar], dim=-1)
        # return: (seq_len, channel=2)

    def __len__(self):
        return len(self.filepths)


################
# LIBRIDATASET #
################
# Librispeech Dataset (works in bucketing style)
# Parameters
#     - file_path    : str, file path to dataset
#     - split        : str, data split (train / dev / test)
#     - max_timestep : int, max len for input (set to 0 for no restriction)
#     - bucket_size  : int, batch size for each bucket
class LibriDataset(Dataset):
    def __init__(self, file_path, sets, bucket_size, max_timestep=0, drop=False):
        # define default length
        self.X = []

        # Read file
        self.root = file_path
        tables = [pd.read_csv(os.path.join(file_path, s + '.csv')) for s in sets]
        self.table = pd.concat(tables, ignore_index=True).sort_values(by=['length'], ascending=False)

        # Crop seqs that are too long
        if drop and max_timestep > 0:
            self.table = self.table[self.table.length < max_timestep]
    
    def __len__(self):
        return len(self.X)


###############
# MEL DATASET #
###############
'''
The Acoustic dataset that loads different types of handcrafted features of the LibriSpeech corpus.
Currently supports 'data/libri_mel160_subword5000' and 'data/libri_fmllr_cmvn' for different preprocessing features.
'''
class AcousticDataset(LibriDataset):
    
    def __init__(self, run_mam, file_path, sets, bucket_size, max_timestep=0, drop=False, mam_config=None):
        super(AcousticDataset, self).__init__(file_path, sets, bucket_size, max_timestep, drop)

        self.run_mam = run_mam
        self.mam_config = mam_config
        self.sample_step = mam_config['max_input_length'] if 'max_input_length' in mam_config else 0
        if self.sample_step > 0: print('[Dataset] - Sampling random segments for training, sample length:', self.sample_step)
        X = self.table['file_path'].tolist()
        X_lens = self.table['length'].tolist()

        # Use bucketing to allow different batch size at run time
        self.X = []
        batch_x, batch_len = [], []

        for x, x_len in zip(X, X_lens):
            batch_x.append(x)
            batch_len.append(x_len)
            
            # Fill in batch_x until batch is full
            if len(batch_x) == bucket_size:
                # Half the batch size if seq too long
                if (bucket_size >= 2) and (max(batch_len) > HALF_BATCHSIZE_TIME):
                    self.X.append(batch_x[:bucket_size//2])
                    self.X.append(batch_x[bucket_size//2:])
                else:
                    self.X.append(batch_x)
                batch_x, batch_len = [], []
        
        # Gather the last batch
        if len(batch_x) > 0:
            self.X.append(batch_x)

    
    def sample(self, x):
        if len(x) < self.sample_step: return x
        idx = random.randint(0, len(x)-self.sample_step)
        return x[idx:idx+self.sample_step]


    def __getitem__(self, index):
        # Load acoustic feature and pad
        if self.sample_step > 0:
            x_batch = [torch.FloatTensor(self.sample(np.load(os.path.join(self.root, x_file)))) for x_file in self.X[index]]
        else:
            x_batch = [torch.FloatTensor(np.load(os.path.join(self.root, x_file))) for x_file in self.X[index]]
        x_pad_batch = pad_sequence(x_batch, batch_first=True)
        if self.run_mam: x_pad_batch = process_train_MAM_data(spec=(x_pad_batch,), config=self.mam_config)
        return x_pad_batch


######################
# MEL LINEAR DATASET #
######################
'''
The LibriSpeech train-clean-360 (Mel Spectrogram, Linear Spectrogram) dataset
'''
class Mel_Linear_Dataset(LibriDataset):
    
    def __init__(self, file_path, target_path, sets, bucket_size, max_timestep=0, drop=False, mam_config=None):
        super(Mel_Linear_Dataset, self).__init__(file_path, sets, bucket_size, max_timestep, drop)

        self.mam_config = mam_config
        # Read Target file
        self.t_root = target_path
        t_tables = [pd.read_csv(os.path.join(target_path, s + '.csv')) for s in sets]
        self.t_table = pd.concat(t_tables, ignore_index=True).sort_values(by=['length'], ascending=False)

        T = self.t_table['file_path'].tolist()
        X = self.table['file_path'].tolist()
        X_lens = self.table['length'].tolist()

        # Use bucketing to allow different batch sizes at run time
        self.T = []
        self.X = []
        batch_t, batch_x, batch_len = [], [], []

        for t, x, x_len in zip(T, X, X_lens):
            batch_t.append(t)
            batch_x.append(x)
            batch_len.append(x_len)
            
            # Fill in batch_x until batch is full
            if len(batch_x) == bucket_size:
                # Half the batch size if seq too long
                if (bucket_size >= 2) and (max(batch_len) > HALF_BATCHSIZE_TIME):
                    self.T.append(batch_t[:bucket_size//2])
                    self.T.append(batch_t[bucket_size//2:])
                    self.X.append(batch_x[:bucket_size//2])
                    self.X.append(batch_x[bucket_size//2:])
                else:
                    self.T.append(batch_t)
                    self.X.append(batch_x)
                batch_t, batch_x, batch_len = [], [], []
        
        # Gather the last batch
        if len(batch_x) > 0:
            self.T.append(batch_t)
            self.X.append(batch_x)

    def __getitem__(self, index):
        # Load acoustic feature and pad
        x_batch = [torch.FloatTensor(np.load(os.path.join(self.root, x_file))) for x_file in self.X[index]]
        x_pad_batch = pad_sequence(x_batch, batch_first=True)
        # Return (x_spec, t_spec)
        t_batch = [torch.FloatTensor(np.load(os.path.join(self.t_root, t_file))) for t_file in self.T[index]]
        t_pad_batch = pad_sequence(t_batch, batch_first=True)
        batch = process_train_MAM_data(spec=(x_pad_batch, t_pad_batch), config=self.mam_config)
        return batch


#####################
# MEL PHONE DATASET #
#####################
'''
The LibriSpeech train-clean-360 (speech, phone) dataset
'''
class Mel_Phone_Dataset(LibriDataset):
    
    def __init__(self, run_mam, file_path, phone_path, sets, bucket_size, max_timestep=0, drop=False, train_proportion=1.0, mam_config=None):
        super(Mel_Phone_Dataset, self).__init__(file_path, sets, bucket_size, max_timestep, drop)

        self.run_mam = run_mam
        self.mam_config = mam_config
        self.phone_path = phone_path
        self.class_num = len(pickle.load(open(os.path.join(phone_path, 'phone2idx.pkl'), 'rb')))
        print('[Dataset] - Possible phone classes: ', self.class_num)

        unaligned = pickle.load(open(os.path.join(phone_path, 'unaligned.pkl'), 'rb'))
        X = self.table['file_path'].tolist()
        X_lens = self.table['length'].tolist()
        if train_proportion < 1.0:
            print('[Dataset] - Truncating dataset size from ', len(X), end='')
            chose_proportion = int(len(X)*train_proportion)
            sample_index = sorted(random.sample(range(len(X)), chose_proportion), reverse=True)
            X = np.asarray(X)[sample_index]
            X_lens = np.asarray(X_lens)[sample_index]
            print(' to ', len(X))
            if len(X) < 200: # is a batch is too small, manually duplicate epoch size to increase dataloader speed.
                for _ in range(4): 
                    X = np.concatenate((X, X), axis=0)
                    X_lens = np.concatenate((X_lens, X_lens), axis=0)
        elif train_proportion > 1.0:
            raise ValueError('Invalid range for `train_proportion`, (0.0, 1.0] is the appropriate range!)')

        # Use bucketing to allow different batch sizes at run time
        self.X = []
        batch_x, batch_len = [], []

        for x, x_len in zip(X, X_lens):
            if x not in unaligned:
                batch_x.append(x)
                batch_len.append(x_len)
                
                # Fill in batch_x until batch is full
                if len(batch_x) == bucket_size:
                    # Half the batch size if seq too long
                    if (bucket_size >= 2) and (max(batch_len) > HALF_BATCHSIZE_TIME):
                        self.X.append(batch_x[:bucket_size//2])
                        self.X.append(batch_x[bucket_size//2:])
                    else:
                        self.X.append(batch_x)
                    batch_x, batch_len = [], []
        
        # Gather the last batch
        if len(batch_x) > 0:
            if x not in unaligned:
                self.X.append(batch_x)

    def match_sequence(self, x_batch, p_batch):
        truncated_length = min(x_batch.shape[1], p_batch.shape[1])
        x_match_batch = x_batch[:, :truncated_length, :]
        p_match_batch = p_batch[:, :truncated_length]
        return x_match_batch, p_match_batch

    def __getitem__(self, index):
        # Load acoustic feature and pad
        x_batch = [torch.FloatTensor(np.load(os.path.join(self.root, x_file))) for x_file in self.X[index]]
        x_pad_batch = pad_sequence(x_batch, batch_first=True)
        p_batch = [torch.LongTensor(pickle.load(open(os.path.join(self.phone_path, \
                   x_file.replace('npy', 'pkl')), "rb"))) for x_file in self.X[index]]
        p_pad_batch = pad_sequence(p_batch, batch_first=True)
        x_match_batch, p_match_batch = self.match_sequence(x_pad_batch, p_pad_batch)
        # Return (x_spec, phone_label)
        if self.run_mam:
            x_match_batch = process_test_MAM_data(spec=(x_match_batch,), config=self.mam_config)
        return x_match_batch, p_match_batch


#####################
# CPC PHONE DATASET #
#####################
'''
The LibriSpeech train-clean-100 (speech, phone) dataset, idendical alignment and split with the CPC paper
'''
class CPC_Phone_Dataset(LibriDataset):
    
    def __init__(self, run_mam, file_path, phone_path, sets, bucket_size, max_timestep=0, drop=False, mam_config=None, split='train', seed=1337):
        super(CPC_Phone_Dataset, self).__init__(file_path, sets, bucket_size, max_timestep, drop)

        assert('train-clean-100' in sets and len(sets) == 1) # `sets` must be ['train-clean-100']
        random.seed(seed)
        self.run_mam = run_mam
        self.mam_config = mam_config
        self.phone_path = phone_path
        phone_file = open(os.path.join(phone_path, 'converted_aligned_phones.txt')).readlines()
        
        self.Y = {}
        # phone_set = []
        for line in phone_file:
            line = line.strip('\n').split(' ')
            self.Y[line[0]] = [int(p) for p in line[1:]]
            # for p in line[1:]: 
                # if p not in phone_set: phone_set.append(p)
        self.class_num = 41 # len(phone_set) # uncomment the above lines if you want to recompute
        
        if split == 'train' or split == 'dev':
            usage_list = open(os.path.join(phone_path, 'train_split.txt')).readlines()
            random.shuffle(usage_list)
            percent = int(len(usage_list)*0.9)
            usage_list = usage_list[:percent] if split == 'train' else usage_list[percent:]
        elif split == 'test':
            usage_list = open(os.path.join(phone_path, 'test_split.txt')).readlines()
        else:
            raise ValueError('Invalid \'split\' argument for dataset: CPC_Phone_Dataset!')
        usage_list = [line.strip('\n') for line in usage_list]
        print('[Dataset] - Possible phone classes: ' + str(self.class_num) + ', number of data: ' + str(len(usage_list)))

        X = self.table['file_path'].tolist()
        X_lens = self.table['length'].tolist()

        # Use bucketing to allow different batch sizes at run time
        self.X = []
        batch_x, batch_len = [], []

        for x, x_len in zip(X, X_lens):
            if self.parse_x_name(x) in usage_list:
                batch_x.append(x)
                batch_len.append(x_len)
                
                # Fill in batch_x until batch is full
                if len(batch_x) == bucket_size:
                    # Half the batch size if seq too long
                    if (bucket_size >= 2) and (max(batch_len) > HALF_BATCHSIZE_TIME):
                        self.X.append(batch_x[:bucket_size//2])
                        self.X.append(batch_x[bucket_size//2:])
                    else:
                        self.X.append(batch_x)
                    batch_x, batch_len = [], []
        
        # Gather the last batch
        if len(batch_x) > 0:
            if self.parse_x_name(x) in usage_list:
                self.X.append(batch_x)

    def parse_x_name(self, x):
        return x.split('/')[-1].split('.')[0]

    def match_sequence(self, x_batch, p_batch):
        truncated_length = min(x_batch.shape[1], p_batch.shape[1])
        x_match_batch = x_batch[:, :truncated_length, :]
        p_match_batch = p_batch[:, :truncated_length]
        return x_match_batch, p_match_batch

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # Load acoustic feature and pad
        x_batch = [torch.FloatTensor(np.load(os.path.join(self.root, x_file))) for x_file in self.X[index]]
        x_pad_batch = pad_sequence(x_batch, batch_first=True)
        p_batch = [torch.LongTensor(self.Y[self.parse_x_name(x_file)]) for x_file in self.X[index]]
        p_pad_batch = pad_sequence(p_batch, batch_first=True)
        x_match_batch, p_match_batch = self.match_sequence(x_pad_batch, p_pad_batch)
        # Return (x_spec, phone_label)
        if self.run_mam:
            x_match_batch = process_test_MAM_data(spec=(x_match_batch,), config=self.mam_config)
        return x_match_batch, p_match_batch


class Mosei_Dataset(Dataset):
    def __init__(self, run_mam, split='train', bucket_size=8, train_proportion=1.0, max_timestep=0, drop=True, mam_config=None, mosei_config=None):
        
        assert(mosei_config is not None), 'MOSEI config is necessary for this dataset'
        self.run_mam = run_mam
        self.mam_config = mam_config
        self.config = mosei_config

        self.csv_path = os.path.join(mosei_config['path'], 'mosei_no_semi.csv')
        self.npy_dir = os.path.join(mosei_config['path'], mosei_config['feature'])
        self.split = split

        if mosei_config['standard_split']:
            raise NotImplementedError('MOSEI standard splits is not supported')
        else:
            all_table = pd.read_csv(self.csv_path)
            starts = all_table.start
            ends = all_table.end
            intervals = ends - starts
            all_table = all_table[intervals <= mosei_config['max_time']]
            all_table = all_table[intervals >= mosei_config['min_time']]
            all_table = all_table[all_table.sentiment.abs() >= mosei_config['sentiment_threshold']]

            if mosei_config['split_by'] == 'segmented':
                train = all_table.sample(frac=mosei_config['split_ratio'], random_state=mosei_config['random_seed'])
                test = all_table.drop(train.index)
            elif mosei_config['split_by'] == 'unsegmented':
                all_filenames = all_table.filename.value_counts().index.values
                all_filenames.sort()
                all_filenames_len = len(all_filenames)
                np.random.seed(mosei_config['random_seed'])
                permute = np.random.permutation(all_filenames_len)
                train_filenames = all_filenames[permute[ : int(mosei_config['split_ratio'] * all_filenames_len)]]
                def judge(filename):
                    if filename in train_filenames:
                        return 'train'
                    else:
                        return 'test'
                all_table['split'] = all_table.filename.apply(judge)
                train = all_table[all_table.split == 'train']
                test = all_table.drop(train.index)
                train = train.sample(frac=train_proportion, random_state=mosei_config['sample_seed'])
            else:
                raise NotImplementedError
            print(f'[DATALOADER] - Training set: {len(train)}')
            print(f'[DATALOADER] - Testing set: {len(test)}')

            if split == 'train':
                self.table = train.sort_values(by=['length'], ascending=False)
            elif split == 'test':
                self.table = test.sort_values(by=['length'], ascending=False)
            else:
                raise NotImplementedError('Invalid `split` argument!')

        if mosei_config['label_mode'] == 'original':
            self.table['label'] = self.table.sentiment.astype(int)  # cause the labels given are average label over all annotaters, so we first round them
            self.table.label += 3  # cause pytorch only accepts non-negative class value, we convert original [-3, -2, -1, 0, 1, 2, 3] into [0, 1, 2, 3, 4, 5, 6]
            self.class_num = 7
        elif mosei_config['label_mode'] == 'positive_negative':
            self.table['label'] = (self.table.sentiment > 0).astype(np.int64)
            self.class_num = 2
        elif mosei_config['label_mode'] == 'regression':
            self.table['label'] = self.table.sentiment
            self.class_num = 1
        else:
            raise NotImplementedError('Not supported label mode')

        # print the majority baseline if is classification task
        if self.class_num > 1:
            value_counts = self.table.label.value_counts()
            majority = value_counts.max()
            all_count = value_counts.sum()
            print(f'[DATALOADER] - Majority: {majority * 1.0 / all_count}')

        # Drop seqs that are too long
        if drop and max_timestep > 0:
            self.table = self.table[self.table.length < max_timestep]

        Y = self.table['label'].tolist()  # (all_data, )
        X = self.table['key'].tolist()
        X = [key + '.npy' for key in X]
        X_lens = self.table['length'].tolist()

        self.Y = []
        self.X = []
        batch_y, batch_x, batch_len = [], [], []

        for y, x, x_len in zip(Y, X, X_lens):
            batch_y.append(y)
            batch_x.append(x)
            batch_len.append(x_len)
            
            # Fill in batch_x until batch is full
            if len(batch_x) == bucket_size:
                # Half the batch size if seq too long
                if (bucket_size >= 2) and (max(batch_len) > HALF_BATCHSIZE_TIME):
                    self.Y.append(batch_y[:bucket_size//2])
                    self.Y.append(batch_y[bucket_size//2:])
                    self.X.append(batch_x[:bucket_size//2])
                    self.X.append(batch_x[bucket_size//2:])
                else:
                    self.Y.append(batch_y)
                    self.X.append(batch_x)
                batch_y, batch_x, batch_len = [], [], []
        
        # Gather the last batch
        if len(batch_x) > 0:
            self.Y.append(batch_y)
            self.X.append(batch_x)

        if split == 'train':
            self.Y *= int(1.0 / train_proportion)
            self.X *= int(1.0 / train_proportion)


    def __getitem__(self, index):
        # Load acoustic feature and pad
        x_batch = [torch.FloatTensor(np.load(os.path.join(self.npy_dir, x_file))) for x_file in self.X[index]]  # [(seq, feature), ...]
        x_pad_batch = pad_sequence(x_batch, batch_first=True)  # (batch, seq, feature) with all seq padded with zeros to align the longest seq in this batch
        truncate_length = self.config['truncate_length']
        if x_pad_batch.size(1) > self.config['truncate_length']:
            x_pad_batch = x_pad_batch[:, :truncate_length, :]

        # Load label
        if self.config['label_mode'] == 'regression':
            y_batch = torch.FloatTensor(self.Y[index])  # (batch, )
        else:
            y_batch = torch.LongTensor(self.Y[index])  # (batch, )
            # y_broadcast_int_batch = y_batch.repeat(x_pad_batch.size(1), 1).T  # (batch, seq)

        if self.run_mam:
            x_pad_batch = process_test_MAM_data(spec=(x_pad_batch,), config=self.mam_config)
        return x_pad_batch, y_batch
    
    def __len__(self):
        return len(self.X)


#######################
# MEL SPEAKER DATASET #
#######################
'''
The LibriSpeech (speech, speaker) dataset
'''
class Speaker_Dataset(Dataset):
    
    def __init__(self, split, run_mam, file_path, sets, bucket_size, split_path=None, max_timestep=0, drop=False, mam_config=None, seed=1337):
        
        random.seed(seed)
        self.run_mam = run_mam
        self.mam_config = mam_config
        self.root = file_path

        # Load the input sets
        tables = [pd.read_csv(os.path.join(file_path, s + '.csv')) for s in sets]
        self.table = pd.concat(tables, ignore_index=True).sort_values(by=['length'], ascending=False)
        X = self.table['file_path'].tolist()
        X_lens = self.table['length'].tolist()

        # Compute speaker dictionary
        print('[Dataset] - Computing speaker class...')
        speakers = self.get_all_speakers(X)
        self.speaker2idx = self.compute_speaker2idx(speakers)
        self.class_num = len(self.speaker2idx)

        # Crop seqs that are too long
        if drop and max_timestep > 0:
            self.table = self.table[self.table.length < max_timestep]
        
        # if using 'train-clean-100' and the cpc split files exist, use them:
        usage_list = []
        if len(sets) == 1 and 'train-clean-100' in sets:
            # use CPC split:
            if (split == 'train' or split == 'dev') and os.path.isfile(os.path.join(split_path, 'train_split.txt')):
                usage_list = open(os.path.join(split_path, 'train_split.txt')).readlines()
                random.shuffle(usage_list)
                percent = int(len(usage_list)*0.9)
                usage_list = usage_list[:percent] if split == 'train' else usage_list[percent:]
            elif split == 'test' and os.path.isfile(os.path.join(split_path, 'test_split.txt')):
                usage_list = open(os.path.join(split_path, 'test_split.txt')).readlines()
            else:
                raise NotImplementedError('Invalid `split` argument!')
            
            self.table = tables
            usage_list = [line.strip('\n') for line in usage_list]
            print('[Dataset] - Using CPC train/test splits.')
            print('[Dataset] - Possible speaker classes: ' + str(self.class_num) + ', number of data: ' + str(len(usage_list)))

        # else use random 8:1:1 split
        if len(usage_list) == 0:
            random.shuffle(X)
            percent_train, percent_dev, percent_test = int(len(X)*0.8), int(len(X)*0.1), int(len(X)*0.1)
            if split == 'train':
                X = X[:percent_train]
            elif split == 'dev':
                X = X[percent_train : percent_train+percent_dev]
            elif split == 'test':
                X = X[-percent_test:]
            else:
                raise NotImplementedError('Invalid `split` argument!')
            print('[Dataset] - Possible speaker classes: ' + str(self.class_num) + ', number of data: ' + str(len(X)))

        # Use bucketing to allow different batch sizes at run time
        self.X = []
        batch_x, batch_len = [], []

        for x, x_len in zip(X, X_lens):
            if len(usage_list) == 0 or self.parse_x_name(x) in usage_list: # check if x is in list if list not empty
                speaker = self.get_speaker_from_path(x)
                if speaker in self.speaker2idx:
                    batch_x.append(x)
                    batch_len.append(x_len)
                    
                    # Fill in batch_x until batch is full
                    if len(batch_x) == bucket_size:
                        # Half the batch size if seq too long
                        if (bucket_size >= 2) and (max(batch_len) > HALF_BATCHSIZE_TIME):
                            self.X.append(batch_x[:bucket_size//2])
                            self.X.append(batch_x[bucket_size//2:])
                        else:
                            self.X.append(batch_x)
                        batch_x, batch_len = [], []
        
        # Gather the last batch
        if len(batch_x) > 0:
            if len(usage_list) == 0 or self.parse_x_name(x) in usage_list: # check if x is in list if list not empty
                self.X.append(batch_x)

    def parse_x_name(self, x):
        return x.split('/')[-1].split('.')[0]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # Load acoustic feature and pad
        x_batch = [torch.FloatTensor(np.load(os.path.join(self.root, x_file))) for x_file in self.X[index]]
        x_pad_batch = pad_sequence(x_batch, batch_first=True)
        # Return (x_spec, speaker_label)
        s_batch = torch.LongTensor([self.speaker2idx[self.get_speaker_from_path(x_file)] for x_file in self.X[index]])
        if self.run_mam:
            x_pad_batch = process_test_MAM_data(spec=(x_pad_batch,), config=self.mam_config)
        return x_pad_batch, s_batch

    def get_speaker_from_path(self, x):
        return x.split('/')[-1].split('.')[0].split('-')[0]

    def get_all_speakers(self, X):
        speaker_set = {}
        for x in X:
            speaker = self.get_speaker_from_path(x)
            if speaker not in speaker_set:
                speaker_set[speaker] = 0
            else:
                speaker_set[speaker] += 1
        return speaker_set

    def compute_speaker2idx(self, speakers):
        idx = 0
        speaker2idx = {}
        for speaker in sorted(speakers):
            if speaker not in speaker2idx and speakers[speaker] > SPEAKER_THRESHOLD: # eliminate the speakers with too few utterance
                speaker2idx[speaker] = idx
                idx += 1
        return speaker2idx


##################
# GET DATALOADER #
##################
def get_Dataloader(split, load, data_path, batch_size, max_timestep, 
                   use_gpu, n_jobs, train_set, dev_set, test_set, dev_batch_size, 
                   target_path=None, phone_path=None, seed=1337,
                   mam_config=None, sentiment_config=None,
                   decode_beam_size=None, run_mam=False, train_proportion=1.0, **kwargs):

    # Decide which split to use: train/dev/test
    if split == 'train':
        bs = batch_size
        shuffle = True
        sets = train_set
        drop_too_long = True
    elif split == 'dev':
        bs = dev_batch_size
        shuffle = False
        sets = dev_set if load != 'cpc_phone' and load != 'speaker' else train_set # the CPC paper uses its own train/test split from train-clean-100
        drop_too_long = True
    elif split == 'test':
        bs = 1 if decode_beam_size is not None else dev_batch_size
        n_jobs = 1
        shuffle = False
        sets = test_set if load != 'cpc_phone' and load != 'speaker' else train_set # the CPC paper uses its own train/test split from train-clean-100
        drop_too_long = False
    else:
        raise NotImplementedError('Unsupported `split` argument: ' + split)

    # Decide which task (or dataset) to propogate through model
    if load == 'acoustic':
        ds = AcousticDataset(run_mam=run_mam, file_path=data_path, sets=sets, max_timestep=max_timestep,
                             bucket_size=bs, drop=drop_too_long, mam_config=mam_config)
    elif load == 'duo':
        assert(target_path is not None), '`target path` must be provided for this dataset.'
        ds = Mel_Linear_Dataset(file_path=data_path, target_path=target_path, sets=sets, max_timestep=max_timestep,
                                bucket_size=bs, drop=drop_too_long, mam_config=mam_config)
    elif load == 'montreal_phone':
        assert(phone_path is not None), '`phone path` must be provided for this dataset.'
        ds = Mel_Phone_Dataset(run_mam=run_mam, file_path=data_path, phone_path=phone_path, sets=sets, max_timestep=max_timestep,
                               bucket_size=bs, drop=drop_too_long, mam_config=mam_config,
                               train_proportion=train_proportion if split != 'test' else 1.0)
    elif load == 'cpc_phone':
        assert(phone_path is not None), '`phone path` must be provided for this dataset.'
        ds = CPC_Phone_Dataset(run_mam=run_mam, file_path=data_path, phone_path=phone_path, sets=sets, max_timestep=max_timestep,
                               bucket_size=bs, drop=drop_too_long, mam_config=mam_config, split=split, seed=seed)
    elif load == 'sentiment':
        assert(sentiment_config is not None), '`sentiment config` must be provided for this dataset.'
        ds = Mosei_Dataset(run_mam=run_mam, split=split, max_timestep=max_timestep, train_proportion=train_proportion,
                           bucket_size=bs, drop=drop_too_long, mam_config=mam_config, mosei_config=sentiment_config['mosei'])
    elif load == 'speaker':
        ds = Speaker_Dataset(split=split, run_mam=run_mam, file_path=data_path, split_path=phone_path, sets=sets, max_timestep=max_timestep,
                             bucket_size=bs, drop=drop_too_long, mam_config=mam_config, seed=seed)
    else:
        raise NotImplementedError('Invalid `load` argument for `get_Dataloader()`!')

    return DataLoader(ds, batch_size=1, shuffle=shuffle, drop_last=False, num_workers=n_jobs, pin_memory=use_gpu)