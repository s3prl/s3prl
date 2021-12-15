# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ pretrain/bucket_dataset.py ]
#   Synopsis     [ the general acoustic dataset with bucketing ]
#   Author1      [ Andy T. Liu (https://github.com/andi611) ]
#   Author2      [ Heng-Jui Chang (https://github.com/vectominist) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import random
import pandas as pd
from torch.utils.data.dataset import Dataset


HALF_BATCHSIZE_TIME = 99999


################
# FEAT DATASET #
################
class FeatDataset(Dataset):
    """Base On-the-fly feature dataset by Andy T. Liu"""
    
    def __init__(self, extracter, task_config, bucket_size, file_path, sets, 
                 max_timestep=0, libri_root=None, **kwargs):
        super(FeatDataset, self).__init__()

        self.extracter = extracter
        self.task_config = task_config
        self.libri_root = libri_root
        self.sample_length = task_config['sequence_length'] 
        if self.sample_length > 0:
            print('[Dataset] - Sampling random segments for training, sample length:', self.sample_length)
        
        # Read file
        self.root = file_path
        tables = [pd.read_csv(os.path.join(file_path, s + '.csv')) for s in sets]
        self.table = pd.concat(tables, ignore_index=True).sort_values(by=['length'], ascending=False)
        print('[Dataset] - Training data from these sets:', str(sets))

        # Drop seqs that are too long
        if max_timestep > 0:
            self.table = self.table[self.table.length < max_timestep]
        # Drop seqs that are too short
        if max_timestep < 0:
            self.table = self.table[self.table.length > (-1 * max_timestep)]

        X = self.table['file_path'].tolist()
        X_lens = self.table['length'].tolist()
        self.num_samples = len(X)
        print('[Dataset] - Number of individual training instances:', self.num_samples)

        # Use bucketing to allow different batch size at run time
        self.X = []
        batch_x, batch_len = [], []

        for x, x_len in zip(X, X_lens):
            batch_x.append(x)
            batch_len.append(x_len)
            
            # Fill in batch_x until batch is full
            if len(batch_x) == bucket_size:
                # Half the batch size if seq too long
                if (bucket_size >= 2) and (max(batch_len) > HALF_BATCHSIZE_TIME) and self.sample_length == 0:
                    self.X.append(batch_x[:bucket_size//2])
                    self.X.append(batch_x[bucket_size//2:])
                else:
                    self.X.append(batch_x)
                batch_x, batch_len = [], []
        
        # Gather the last batch
        if len(batch_x) > 1: 
            self.X.append(batch_x)

    def _sample(self, x):
        if self.sample_length <= 0: return x
        if len(x) < self.sample_length: return x
        idx = random.randint(0, len(x)-self.sample_length)
        return x[idx:idx+self.sample_length]

    def __len__(self):
        return len(self.X)

    def collate_fn(self, items):
        items = items[0] # hack bucketing
        return items


################
# WAVE DATASET #
################
class WaveDataset(Dataset):
    """Base waveform dataset for Disiller by Heng-Jui Chang"""

    def __init__(
        self,
        task_config,
        bucket_size,
        file_path,
        sets,
        max_timestep=0,
        libri_root=None,
        **kwargs
    ):
        super().__init__()

        self.task_config = task_config
        self.libri_root = libri_root
        self.sample_length = task_config["sequence_length"]
        if self.sample_length > 0:
            print(
                "[Dataset] - Sampling random segments for training, sample length:",
                self.sample_length,
            )

        # Read file
        self.root = file_path
        tables = [pd.read_csv(os.path.join(file_path, s + ".csv")) for s in sets]
        self.table = pd.concat(tables, ignore_index=True).sort_values(
            by=["length"], ascending=False
        )
        print("[Dataset] - Training data from these sets:", str(sets))

        # Drop seqs that are too long
        if max_timestep > 0:
            self.table = self.table[self.table.length < max_timestep]
        # Drop seqs that are too short
        if max_timestep < 0:
            self.table = self.table[self.table.length > (-1 * max_timestep)]

        X = self.table["file_path"].tolist()
        X_lens = self.table["length"].tolist()
        self.num_samples = len(X)
        print("[Dataset] - Number of individual training instances:", self.num_samples)

        # Use bucketing to allow different batch size at run time
        self.X = []
        batch_x, batch_len = [], []

        for x, x_len in zip(X, X_lens):
            batch_x.append(x)
            batch_len.append(x_len)

            # Fill in batch_x until batch is full
            if len(batch_x) == bucket_size:
                # Half the batch size if seq too long
                if (
                    (bucket_size >= 2)
                    and (max(batch_len) > HALF_BATCHSIZE_TIME)
                    and self.sample_length == 0
                ):
                    self.X.append(batch_x[: bucket_size // 2])
                    self.X.append(batch_x[bucket_size // 2 :])
                else:
                    self.X.append(batch_x)
                batch_x, batch_len = [], []

        # Gather the last batch
        if len(batch_x) > 1:
            self.X.append(batch_x)

    def _sample(self, x):
        if self.sample_length <= 0:
            return x
        if len(x) < self.sample_length:
            return x
        idx = random.randint(0, len(x) - self.sample_length)
        return x[idx : idx + self.sample_length]

    def __len__(self):
        return len(self.X)

    def collate_fn(self, items):
        items = items[0]  # hack bucketing
        assert (
            len(items) == 4
        ), "__getitem__ should return (wave_input, wave_orig, wave_len, pad_mask)"
        return items