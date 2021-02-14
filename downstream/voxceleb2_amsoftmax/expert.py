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
import numpy as np
#-------------#
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
#-------------#
from .model import Model, AdMSoftmaxLoss
from .dataset import SpeakerVerifi_train, SpeakerVerifi_dev, SpeakerVerifi_test
from argparse import Namespace
from .utils import EER, compute_metrics
import IPython
import pdb


class DownstreamExpert(nn.Module):
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log

    Note 1.
        dataloaders should output in the following format:

        [[wav1, wav2, ...], your_other_contents, ...]

        where wav1, wav2 ... are in variable length
        and wav1 is in torch.FloatTensor
    """

    def __init__(self, upstream_dim, downstream_expert, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.downstream = downstream_expert
        self.datarc = downstream_expert['datarc']
        self.modelrc = downstream_expert['modelrc']

        self.train_dataset = SpeakerVerifi_train(self.datarc['train']['file_path'], self.datarc['train']['meta_data'], self.datarc['train']['max_timestep'])
        self.dev_dataset = SpeakerVerifi_dev(self.datarc['dev']['file_path'], self.datarc['dev']['meta_data'])
        self.test_dataset = SpeakerVerifi_test(self.datarc['test']['file_path'], self.datarc['test']['meta_data'])
        
        self.connector = nn.Linear(self.upstream_dim, self.modelrc['input_dim'])

        self.model = Model(input_dim=self.modelrc['agg_dim'], agg_module=self.modelrc['agg_module'], config=self.modelrc)
        self.objective = AdMSoftmaxLoss(self.modelrc['agg_dim'], self.train_dataset.speaker_num, s=30.0, m=0.4)

        self.score_fn  = nn.CosineSimilarity(dim=-1)
        self.eval_metric = EER

    # Interface
    def get_dataloader(self, mode):
        """
        Args:
            mode: string
                'train', 'dev' or 'test'

        Return:
            a torch.utils.data.DataLoader returning each batch in the format of:

            [wav1, wav2, ...], your_other_contents1, your_other_contents2, ...

            where wav1, wav2 ... are in variable length
            each wav is torch.FloatTensor in cpu with:
                1. dim() == 1
                2. sample_rate == 16000
                3. directly loaded by torchaudio
        """

        if mode == 'train':
            return self._get_train_dataloader(self.train_dataset)            
        elif mode == 'dev':
            return self._get_eval_dataloader(self.dev_dataset)
        elif mode == 'test':
            return self._get_eval_dataloader(self.test_dataset)

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
    def forward(self, mode, features, your_other_contents1, records, **kwargs):
        """
        Args:
            features:
                the features extracted by upstream
                put in the device assigned by command-line args

            labels:
                the frame-wise phone labels

            records:
                defaultdict(list), by appending scalars into records,
                these scalars will be averaged and logged on Tensorboard

            logger:
                Tensorboard SummaryWriter, given here for logging/debugging
                convenience, please use "self.downstream/your_content_name" as key
                name to log your customized contents

            global_step:
                global_step in runner, which is helpful for Tensorboard logging

        Return:
            loss:
                the loss to be optimized, should not be detached
        """
        features_pad = pad_sequence(features, batch_first=True)
        
        attention_mask = [torch.ones((feature.shape[0])) for feature in features] 

        attention_mask_pad = pad_sequence(attention_mask,batch_first=True)

        attention_mask_pad = (1.0 - attention_mask_pad) * -100000.0

        features_pad = self.connector(features_pad)
        agg_vec = self.model(features_pad, attention_mask_pad.cuda())

        if self.training:

            labels = torch.LongTensor(labels).to(features_pad.device)
            loss = self.objective(agg_vec, labels)
            
            return loss

        else:
            # normalize to unit vector 
            agg_vec = agg_vec / (torch.norm(agg_vec, dim=-1).unsqueeze(-1))

            vec1, vec2 = self.separate_data(agg_vec, labels)
            scores = self.score_fn(vec1,vec2).squeeze().cpu().detach().tolist()
            ylabels = torch.stack(labels).cpu().detach().long().tolist()

            if len(ylabels) > 1:
                records['scores'].extend(scores)
                records['ylabels'].extend(ylabels)
            else:
                records['scores'].append(scores)
                records['ylabels'].append(ylabels)

            return torch.tensor(0)

    # interface
    def log_records(self, mode, records, logger, global_step, batch_ids, total_batch_num, **kwargs):
        """
        Args:
            records:
                defaultdict(list), contents already appended

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
        if not self.training:

            EER_result =self.eval_metric(np.array(records['ylabels']), np.array(records['scores']))

            records['EER'] = EER_result[0]

            logger.add_scalar(
                f'{prefix}'+'EER',
                records['EER'],
                global_step=global_step
            )
        
    def separate_data(self, agg_vec, ylabel):

        total_num = len(ylabel) 
        feature1 = agg_vec[:total_num]
        feature2 = agg_vec[total_num:]
        
        return feature1, feature2
