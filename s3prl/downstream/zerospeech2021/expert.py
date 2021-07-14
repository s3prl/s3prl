import numpy as np
import os
import torch

import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from .model import Model
from .dataset import *

import pandas as pd
import sys

class DownstreamExpert(nn.Module):
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log
    """

    def __init__(self, upstream_dim, downstream_expert, **kwargs):
        """
        Args:
            upstream_dim: int
                Different upstream will give different representation dimension
                You might want to first project them to the same dimension
            
            downstream_expert: dict
                The 'downstream_expert' field specified in your downstream config file
                eg. downstream/downstream/example/config.yaml

            **kwargs: dict
                The arguments specified by the argparser in run_downstream.py
                in case you need it.
        """

        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.datarc = downstream_expert['datarc']
        self.modelrc = downstream_expert['modelrc']
        
        self.subdir = downstream_expert['subdir']
        if not os.path.exists(self.subdir):
            os.makedirs(self.subdir)

        self.task = self.datarc['task']
        assert self.task in ['phonetic', 'lexical', 'semantic', 'syntactic']

        self.split = self.datarc[self.task]['split']
        assert self.split in ['dev', 'test']

        if not os.path.exists(os.path.join(self.subdir, self.task)):
            os.makedirs(os.path.join(self.subdir, self.task))

        if self.task == 'phonetic':
            self.portion = self.datarc[self.task][self.split]
            assert self.portion in ['clean', 'other']

            if not os.path.exists(os.path.join(self.subdir, self.task, self.split + '-' + self.portion)):
                os.makedirs(os.path.join(self.subdir, self.task, self.split + '-' + self.portion))

            self.test_dataset = Phonetic(self.split, self.portion, self.datarc['data_dir'])
            self.connector = nn.Linear(upstream_dim, self.modelrc[self.task]['feature_dim'])

        elif self.task == 'lexical':
            pass

        elif self.task == 'semantic':
            self.portion = self.datarc[self.task][self.split]
            assert self.portion in ['librispeech', 'synthetic']

            if not os.path.exists(os.path.join(self.subdir, self.task, self.split)):
                os.makedirs(os.path.join(self.subdir, self.task, self.split))
            if not os.path.exists(os.path.join(self.subdir, self.task, self.split, self.portion)):
                os.makedirs(os.path.join(self.subdir, self.task, self.split, self.portion))

            self.test_dataset = Semantic(self.split, self.portion, self.datarc['data_dir'])
            self.connector = nn.Linear(upstream_dim, self.modelrc[self.task]['feature_dim'])

        elif self.task == 'syntactic':
            pass 

        '''
        self.model = Model(
            output_class_num=self.datarc['num_class'],
            **self.modelrc
        )
        self.objective = nn.CrossEntropyLoss()
        '''

    def _get_train_dataloader(self, dataset):
        return DataLoader(
            dataset, batch_size=self.datarc['train_batch_size'],
            shuffle=True, num_workers=self.datarc['num_workers'],
            collate_fn=dataset.collate_fn
        )

    def _get_eval_dataloader(self, dataset):
        return DataLoader(
            dataset, batch_size=self.datarc['eval_batch_size'],
            shuffle=False, num_workers=self.datarc['num_workers'],
            collate_fn=dataset.collate_fn
        )

    """
    Datalaoder Specs:
        Each dataloader should output a list in the following format:

        [[wav1, wav2, ...], your_other_contents1, your_other_contents2, ...]

        where wav1, wav2 ... are in variable length
        each wav is torch.FloatTensor in cpu with:
            1. dim() == 1
            2. sample_rate == 16000
            3. directly loaded by torchaudio without any preprocessing
    """

    # Interface
    def get_train_dataloader(self):
        return self._get_train_dataloader(self.train_dataset)

    # Interface
    def get_dev_dataloader(self):
        return self._get_eval_dataloader(self.dev_dataset)

    # Interface
    def get_test_dataloader(self):
        return self._get_eval_dataloader(self.test_dataset)

    # Interface
    def forward(self, features, your_other_contents1, records, logger, prefix, global_step, **kwargs):
        """
        This function will be used in both train/dev/test, you can use
        self.training (bool) to control the different behavior for
        training or evaluation (dev/test)

        Args:
            features:
                list of unpadded features [feat1, feat2, ...]
                each feat is in torch.FloatTensor and already
                put in the device assigned by command-line args

            your_other_contents1, ... :
                in the order defined by your dataloader (dataset + collate_fn)
                these are all in cpu, and you can move them to the same device
                as features

            records:
                defaultdict(list), by dumping contents into records,
                these contents can be averaged and logged on Tensorboard
                later by self.log_records

                Note1. downstream/runner.py will call self.log_records
                    1. every log_step during training
                    2. once after evalute the whole dev/test dataloader

                Note2. log_step is defined in your downstream config

            logger:
                Tensorboard SummaryWriter, given here for logging/debugging convenience
                please use f'{prefix}your_content_name' as key name
                to log your customized contents

            prefix:
                used to indicate downstream and train/test on Tensorboard
                eg. 'phone/train-'

            global_step:
                global_step in runner, which is helpful for Tensorboard logging

        Return:
            loss:
                the loss to be optimized, should not be detached
                a single scalar in torch.FloatTensor
        """
        
        if self.task == 'phonetic':
            features = pad_sequence(features, batch_first=True)
            features = self.connector(features)
            
            input_paths = your_other_contents1
            output_paths = [os.path.join(self.subdir, '/'.join(path.split('/')[-3:]))[:-4] + '.txt' for path in input_paths]

            for path, feature in zip(output_paths, features):
                if self.modelrc[self.task]['precision'] == 'None':
                    np.savetxt(path, np.array(feature.cpu()))
                else:
                    np.savetxt(path, np.array(feature.cpu()), fmt='%1.'+str(self.modelrc[self.task]['precision'])+'e')

            return torch.tensor(0)

        elif self.task == 'lexical':
            pass

        elif self.task == 'semantic':
            features = pad_sequence(features, batch_first=True)
            features = self.connector(features)
            
            input_paths = your_other_contents1
            output_paths = [os.path.join(self.subdir, '/'.join(path.split('/')[-4:]))[:-4] + '.txt' for path in input_paths]

            for path, feature in zip(output_paths, features):
                if self.modelrc[self.task]['precision'] == 'None':
                    np.savetxt(path, np.array(feature.cpu()))
                else:
                    np.savetxt(path, np.array(feature.cpu()), fmt='%1.'+str(self.modelrc[self.task]['precision'])+'e')

            return torch.tensor(0)

        elif self.task == 'syntactic':
            pass 

        '''
        utterance_labels = your_other_contents1
        labels = torch.LongTensor(utterance_labels).to(features.device)
        loss = self.objective(predicted, labels)
        predicted_classid = predicted.max(dim=-1).indices
        
        records['acc'] += (predicted_classid == labels).view(-1).cpu().float().tolist()

        if not self.training:
            # some evaluation-only processing, eg. decoding
            pass

        return loss
        '''
        

    # interface
    def log_records(self, records, logger, prefix, global_step, **kwargs):
        """
        This function will be used in both train/dev/test, you can use
        self.training (bool) to control the different behavior for
        training or evaluation (dev/test)

        Args:
            records:
                defaultdict(list), contents already prepared by self.forward

            logger:
                Tensorboard SummaryWriter
                please use f'{prefix}your_content_name' as key name
                to log your customized contents

            prefix:
                used to indicate downstream and train/test on Tensorboard
                eg. 'phone/train-'

            global_step:
                global_step in runner, which is helpful for Tensorboard logging
        """
        '''
        for key, values in records.items():
            average = torch.FloatTensor(values).mean().item()
            logger.add_scalar(
                f'{prefix}{key}',
                average,
                global_step=global_step
            )

        if not self.training:
            # some evaluation-only processing, eg. decoding
            pass
        '''
