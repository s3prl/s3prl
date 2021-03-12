# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ expert.py ]
#   Synopsis     [ the phone linear downstream wrapper ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import math
import torch
import random
import pathlib
#-------------#
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
#-------------#
from .model import Model
from .dataset import SpeakerClassifiDataset
from argparse import Namespace
from pathlib import Path


class DownstreamExpert(nn.Module):
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log
    """

    def __init__(self, upstream_dim, downstream_expert, expdir, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.downstream = downstream_expert
        self.datarc = downstream_expert['datarc']
        self.modelrc = downstream_expert['modelrc']

        root_dir = Path(self.datarc['file_path'])

        self.train_dataset = SpeakerClassifiDataset('train', root_dir, self.datarc['meta_data'], self.datarc['max_timestep'])
        self.dev_dataset = SpeakerClassifiDataset('dev', root_dir, self.datarc['meta_data'])
        self.test_dataset = SpeakerClassifiDataset('test', root_dir, self.datarc['meta_data'])
        
        self.connector = nn.Linear(self.upstream_dim, self.modelrc['input_dim'])

        self.model = Model(input_dim=self.modelrc['input_dim'], agg_module=self.modelrc['agg_module'],output_class_num=self.train_dataset.speaker_num, config=self.modelrc)
        self.objective = nn.CrossEntropyLoss()
        
        self.logging = os.path.join(expdir, 'log.log')
        self.register_buffer('best_score', torch.zeros(1))

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

    def get_train_dataloader(self):
        return self._get_train_dataloader(self.train_dataset)

    def get_dev_dataloader(self):
        return self._get_eval_dataloader(self.dev_dataset)

    def get_test_dataloader(self):
        return self._get_eval_dataloader(self.test_dataset)

    # Interface
    def get_dataloader(self, mode):
        return eval(f'self.get_{mode}_dataloader')()

    # Interface
    def forward(self, mode, features, lengths, labels, records, **kwargs):
        features_pad = pad_sequence(features, batch_first=True)
        
        attention_mask = [torch.ones((feature.shape[0])) for feature in features] 

        attention_mask_pad = pad_sequence(attention_mask,batch_first=True)

        attention_mask_pad = (1.0 - attention_mask_pad) * -100000.0

        features_pad = self.connector(features_pad)
        predicted = self.model(features_pad, attention_mask_pad.cuda())

        labels = torch.LongTensor(labels).to(features_pad.device)
        loss = self.objective(predicted, labels)

        predicted_classid = predicted.max(dim=-1).indices
        records['acc'] += (predicted_classid == labels).view(-1).cpu().float().tolist()
        records['loss'].append(loss.item())

        return loss

    # interface
    def log_records(self, mode, records, logger, global_step, **kwargs):
        save_names = []
        for key, values in records.items():
            average = torch.FloatTensor(values).mean().item()
            logger.add_scalar(
                f'voxceleb1/{mode}-{key}',
                average,
                global_step=global_step
            )
            with open(self.logging, 'a') as f:
                if key == 'acc':
                    f.write(f'{mode} at step {global_step}: {average}\n')
                    if mode == 'dev' and average > self.best_score:
                        self.best_score = torch.ones(1) * average
                        f.write(f'New best on {mode} at step {global_step}: {average}\n')
                        save_names.append(f'{mode}-best.ckpt')
        return save_names
