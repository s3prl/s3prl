# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ expert.py ]
#   Synopsis     [ the phone linear downstream wrapper ]
#   Author       [ Andy T. Liu (https://github.com/andi611) ]
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
from torch.nn.utils.rnn import pad_sequence
#-------------#
from .model import Model
from .dataset import PhoneDataset


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

        self.train_dataset = PhoneDataset('train', self.datarc['train_batch_size'], **self.datarc)
        self.dev_dataset = PhoneDataset('dev', self.datarc['eval_batch_size'], **self.datarc)
        self.test_dataset = PhoneDataset('test', self.datarc['eval_batch_size'], **self.datarc)

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

    def _tile_representations(self, reps, factor):
        """ 
        Tile up the representations by `factor`.
        Input - sequence of representations, shape: (batch_size, seq_len, feature_dim)
        Output - sequence of tiled representations, shape: (batch_size, seq_len * factor, feature_dim)
        """
        assert len(reps.shape) == 3, 'Input argument `reps` has invalid shape: {}'.format(reps.shape)
        tiled_reps = reps.repeat(1, 1, factor)
        tiled_reps = tiled_reps.reshape(reps.size(0), reps.size(1)*factor, reps.size(2))
        return tiled_reps

    def _match_length(self, inputs, labels):
        """
        Since the upstream extraction process can sometimes cause a mismatch
        between the seq lenth of inputs and labels:
        - if len(inputs) > len(labels), we truncate the final few timestamp of inputs to match the length of labels
        - if len(inputs) < len(labels), we duplicate the last timestep of inputs to match the length of labels
        Note that the length of labels should never be changed.
        """
        input_len, label_len = inputs.size(1), labels.size(-1)

        factor = int(round(label_len / input_len))
        if factor > 1:
            inputs = self._tile_representations(inputs, factor)
            input_len = inputs.size(1)

        if input_len > label_len:
            inputs = inputs[:, :label_len, :]
        elif input_len < label_len:
            pad_vec = inputs[:, -1, :].unsqueeze(1) # (batch_size, 1, feature_dim)
            inputs = torch.cat((inputs, pad_vec.repeat(1, label_len-input_len, 1)), dim=1) # (batch_size, seq_len, feature_dim), where seq_len == labels.size(-1)
        return inputs, labels

    # Interface
    def forward(self, mode, features, labels, records, **kwargs):
        """
        Args:
            features:
                list of unpadded features [feat1, feat2, ...]
                each feat is in torch.FloatTensor and already
                put in the device assigned by command-line args

            labels:
                the frame-wise phone labels

            records:
                defaultdict(list), by appending contents into records,
                these contents can be averaged and logged on Tensorboard
                later by self.log_records every log_step

        Return:
            loss:
                the loss to be optimized, should not be detached
        """
        lengths = torch.LongTensor([len(l) for l in labels])

        features = pad_sequence(features, batch_first=True)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100).to(features.device)

        features, labels = self._match_length(features, labels)
        predicted = self.model(features)

        # cause logits are in (batch, seq, class) and labels are in (batch, seq)
        # nn.CrossEntropyLoss expect to have (N, class) and (N,) as input
        # here we flatten logits and labels in order to apply nn.CrossEntropyLoss
        class_num = predicted.size(-1)
        loss = self.objective(predicted.reshape(-1, class_num), labels.reshape(-1))

        predicted_classid = predicted.max(dim=-1).indices
        sames = (predicted_classid == labels)
        for s, l in zip(sames, lengths):
            utter_result = s[:l].tolist()
            records['acc'] += utter_result
            records['sample_wise_metric'] += [torch.FloatTensor(utter_result).mean().item()]

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
        prefix = f'libri_phone/{mode}-'
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