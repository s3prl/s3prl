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
from .model import Model, AdMSoftmaxLoss, UtteranceModel
from .dataset import SpeakerVerifi_train, SpeakerVerifi_dev, SpeakerVerifi_test
from argparse import Namespace
from .utils import EER, compute_metrics


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

        self.train_dataset = SpeakerVerifi_train(self.datarc['vad_config'], **self.datarc['train'])
        self.dev_dataset = SpeakerVerifi_dev(self.datarc['vad_config'], self.datarc["segment_config"], **self.datarc['dev'])
        self.test_dataset = SpeakerVerifi_test(self.datarc['vad_config'],self.datarc["segment_config"], **self.datarc['test'])
        
        self.connector = nn.Linear(self.upstream_dim, self.modelrc['input_dim'])
        self.model = Model(input_dim=self.modelrc['input_dim'], agg_dim=self.modelrc['agg_dim'], agg_module=self.modelrc['agg_module'], config=self.modelrc)
        self.objective = AdMSoftmaxLoss(self.modelrc['input_dim'], self.train_dataset.speaker_num, s=30.0, m=0.4)

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
    def forward(self, mode, features, labels, pair_list, utterid_list, seg_num_list,  records, **kwargs):
        """
        Args:
            features:
                the features extracted by upstream
                put in the device assigned by command-line args

            labels:
                the speaker labels

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
        
        if self.modelrc['module'] == "XVector":
            attention_mask = [torch.ones((feature.shape[0]-14)) for feature in features]
        else:
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

            if len(labels) >1:

                agg_vec_list = [vec for vec in agg_vec]
                for index in range(len(agg_vec)):
                    records[f'utterid_{utterid_list[index]}'].append(agg_vec[index])
                    records[f'utterid_info'].append(f'utterid_{utterid_list[index]}')
                    records[f'pairid_info'].append(f'pairid_{pair_list[index]}')
                    records[f'pairid_{pair_list[index]}'].append(f'utterid_{utterid_list[index]}')
                    records[f'pairid_{pair_list[index]}_label'].append(labels[index])
         
            else:
                records[f'utterid_{utterid_list[0]}'].append(agg_vec[0])
                records[f'utterid_info'].append(f'utterid_{utterid_list[0]}')
                records[f'pairid_info'].append(f'pairid_{pair_list[0]}')
                records[f'pairid_{pair_list[0]}'].append(f'utterid_{utterid_list[0]}')
                records[f'pairid_{pair_list[0]}_label'].append(labels[0])


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
            records = self.declutter(records)

            EER_result =self.eval_metric(np.array(records['ylabels']), np.array(records['scores']))

            records['EER'] = EER_result[0]

            logger.add_scalar(
                f'{mode}-'+'EER',
                records['EER'],
                global_step=global_step
            )
    def declutter(self, records):
        utterance_ids = set(records['utterid_info'])
        for index in utterance_ids:
            records[index] = torch.mean(torch.stack(records[index]),dim=0)
        pair_ids = set(records['pairid_info'])
        for index in pair_ids:
            wav_set = list(set(records[index]))
            if len(wav_set) == 1:
                # wavs1 = records[wav_set[0]][None,:]
                # wavs2 = records[wav_set[0]][:,None]                
                wavs1 = records[wav_set[0]]
                wavs2 = records[wav_set[0]]
            else:
                # wavs1 = records[wav_set[0]][None,:]
                # wavs2 = records[wav_set[1]][:,None]
                wavs1 = records[wav_set[0]]
                wavs2 = records[wav_set[1]]
            # score = torch.mean(self.score_fn(wavs1,wavs2)).squeeze().cpu().detach().tolist()
            score = self.score_fn(wavs1,wavs2).squeeze().cpu().detach().tolist()
            ylabel = list(set(records[f"{index}_label"]))[0]
            records['ylabels'].append(ylabel)
            records['scores'].append(score)
        return records

        
