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
#-------------#
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
#-------------#
from benchmark.downstream.speaker_verifi.model import Model
from benchmark.downstream.speaker_verifi.dataset import SpeakerVerifi_train, SpeakerVerifi_test, SpeakerVerifi_dev
from benchmark.downstream.speaker_verifi.model import GE2E
from benchmark.downstream.speaker_verifi.utils import EER, compute_metrics

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

    def __init__(self, upstream_dim, downstream='speaker_classifi', datarc={}, modelrc={}, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.downstream = downstream
        self.datarc = datarc
        self.modelrc = modelrc

        self.train_dataset = SpeakerVerifi_train(datarc['train']['file_path'], datarc['train']['max_timestep'],datarc['train']['utter_number'])
        self.dev_dataset = SpeakerVerifi_dev(datarc['dev']['file_path'], datarc['dev']['max_timestep'], datarc['dev']['meta_data'])
        self.test_dataset = SpeakerVerifi_test(datarc['test']['file_path'], datarc['test']['meta_data'])
        
        self.connector = nn.Linear(upstream_dim, modelrc['input_dim'])

        self.model = Model(input_dim=modelrc['input_dim'], agg_module=self.modelrc['agg_module'])
        self.objective = GE2E()
        self.score_fn  = nn.CosineSimilarity(dim=-1)
        self.eval_metric = EER

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
    def forward(self, features, lengths, labels,
                records=None, logger=None, prefix=None, global_step=0):
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

        # normalize to unit vector 
        agg_vec = agg_vec / torch.norm(agg_vec, dim=-1).unsqueeze(-1)

        if self.training:
            
            GE2E_matrix = agg_vec.reshape(-1, self.train_dataset.utter_number, agg_vec.shape[-1])
            loss = self.objective(GE2E_matrix)
            
            return loss
        
        else:
            vec1, vec2 = self.separate_data(agg_vec, labels)
            scores = self.score_fn(vec1,vec2).squeeze().cpu().detach().tolist()
            ylabels = torch.stack(labels).cpu().detach().long().tolist()

            if len(ylabels) > 1:
                records['scores'].extends(scores)
                records['ylabels'].extends(ylabels)
            else:
                records['scores'].append(scores)
                records['ylabels'].append(ylabels)

            return torch.tensor(0)

        # interface
    def log_records(self, records, logger, prefix, global_step):
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