# -*- coding: utf-8 -*- #
"""
    FileName     [ dataset.py ]
    Synopsis     [ the emotion classifier dataset ]
    Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""

import json
import random
import logging
from pathlib import Path
from collections import defaultdict
from os.path import join as path_join

import torchaudio
from torch.utils.data import Dataset, Subset
from torchaudio.transforms import Resample

SAMPLE_RATE = 16000
log = logging.getLogger(__name__)

class IEMOCAPDataset(Dataset):
    def __init__(self, data_dir, meta_path, pre_load=True):
        self.data_dir = data_dir
        self.pre_load = pre_load
        with open(meta_path, 'r') as f:
            self.data = json.load(f)
        self.class_dict = self.data['labels']
        self.idx2emotion = {value: key for key, value in self.class_dict.items()}
        self.class_num = len(self.class_dict)
        self.meta_data = self.data['meta_data']
        _, origin_sr = torchaudio.load(
            path_join(self.data_dir, self.meta_data[0]['path']))
        self.resampler = Resample(origin_sr, SAMPLE_RATE)
        if self.pre_load:
            self.wavs = self._load_all()

    @classmethod
    def from_subset(cls, subset: Subset, n_shot: int = None, seed=0):
        random.seed(seed)
        dataset = subset.dataset
        indices = subset.indices
        dataset.meta_data = [dataset.meta_data[idx] for idx in indices]

        if isinstance(n_shot, int):
            emotion2indices = defaultdict(list)
            for metadata in dataset.meta_data:
                emotion2indices[metadata['label']].append(metadata)
            for key in list(emotion2indices.keys()):
                emotion2indices[key] = random.sample(emotion2indices[key], k=n_shot)

            dataset.meta_data = []
            for meta_data_list in emotion2indices.values():
                dataset.meta_data += meta_data_list

        if dataset.pre_load:
            dataset.wavs = dataset._load_all()

        return dataset

    def _load_wav(self, path):
        wav, _ = torchaudio.load(path_join(self.data_dir, path))
        wav = self.resampler(wav).squeeze(0)
        return wav

    def _load_all(self):
        wavforms = []
        for info in self.meta_data:
            wav = self._load_wav(info['path'])
            wavforms.append(wav)
        return wavforms

    def __getitem__(self, idx):
        label = self.meta_data[idx]['label']
        label = self.class_dict[label]
        if self.pre_load:
            wav = self.wavs[idx]
        else:
            wav = self._load_wav(self.meta_data[idx]['path'])
        return wav.numpy(), label, Path(self.meta_data[idx]['path']).stem

    def __len__(self):
        return len(self.meta_data)

def collate_fn(samples):
    return zip(*samples)
