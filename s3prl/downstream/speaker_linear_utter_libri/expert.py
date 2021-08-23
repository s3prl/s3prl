# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ expert.py ]
#   Synopsis     [ the speaker linear downstream wrapper ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import math
import random
from collections import defaultdict
#-------------#
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
#-------------#
from .model import Model
from .dataset import SpeakerDataset


class DownstreamExpert(nn.Module):
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log
    """

    def __init__(self, upstream_dim, downstream_expert, expdir, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.datarc = downstream_expert['datarc']
        self.modelrc = downstream_expert['modelrc']

        self.train_dataset = SpeakerDataset('train', self.datarc['train_batch_size'], **self.datarc)
        self.dev_dataset = SpeakerDataset('dev', self.datarc['eval_batch_size'], **self.datarc)
        self.test_dataset = SpeakerDataset('test', self.datarc['eval_batch_size'], **self.datarc)

        self.model = Model(input_dim=self.upstream_dim, output_class_num=self.train_dataset.class_num, **self.modelrc)
        self.objective = nn.CrossEntropyLoss()

        self.logging = os.path.join(expdir, 'log.log')
        self.best = defaultdict(lambda: 0)

    def _get_train_dataloader(self, dataset):
        return DataLoader(
            dataset, batch_size=1, # for bucketing
            shuffle=True, num_workers=self.datarc['num_workers'],
            drop_last=False, pin_memory=True, collate_fn=dataset.collate_fn
        )

    def _get_eval_dataloader(self, dataset):
        return DataLoader(
            dataset, batch_size=1, # for bucketing
            shuffle=False, num_workers=self.datarc['num_workers'],
            drop_last=False, pin_memory=True, collate_fn=dataset.collate_fn
        )

    """
    Datalaoder Specs:
        Each dataloader should output in the following format:

        [[wav1, wav2, ...], your_other_contents1, your_other_contents2, ...]

        where wav1, wav2 ... are in variable length
        each wav is torch.FloatTensor in cpu with dim()==1 and sample_rate==16000
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
    def get_dataloader(self, mode):
        return eval(f'self.get_{mode}_dataloader')()

    # Interface
    def forward(self, mode, features, labels, records, **kwargs):
        """
        Args:
            features:
                list of unpadded features [feat1, feat2, ...]
                each feat is in torch.FloatTensor and already
                put in the device assigned by command-line args

            labels:
                the utterance-wise spekaer labels

            records:
                defaultdict(list), by appending contents into records,
                these contents can be averaged and logged on Tensorboard
                later by self.log_records every log_step

        Return:
            loss:
                the loss to be optimized, should not be detached
        """

        features = torch.stack([f.mean(dim=0) for f in features], dim=0) # (batch_size, seq_len, feature_dim) -> (batch_size, feature_dim)
        labels = labels.to(features.device)

        predicted = self.model(features)
        loss = self.objective(predicted, labels)

        predicted_classid = predicted.max(dim=-1).indices
        records['acc'] += (predicted_classid == labels).view(-1).cpu().float().tolist()

        return loss

    # interface
    def log_records(self, mode, records, logger, global_step, **kwargs):
        """
        Args:
            records:
                defaultdict(list), contents already appended

            logger:
                Tensorboard SummaryWriter
                please use f'{prefix}your_content_name' as key name
                to log your customized contents

            global_step:
                global_step in runner, which is helpful for Tensorboard logging
        """
        prefix = f'libri_speaker/{mode}-'
        average = torch.FloatTensor(records['acc']).mean().item()
        
        logger.add_scalar(
            f'{prefix}acc',
            average,
            global_step=global_step
        )
        message = f'{prefix}|step:{global_step}|acc:{average}\n'
        save_ckpt = []
        if average > self.best[prefix]:
            self.best[prefix] = average
            message = f'best|{message}'
            name = prefix.split('/')[-1].split('-')[0]
            save_ckpt.append(f'best-states-{name}.ckpt')
        with open(self.logging, 'a') as f:
            f.write(message)
        print(message)

        return save_ckpt
