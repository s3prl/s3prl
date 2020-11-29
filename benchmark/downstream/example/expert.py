import os
import math
import torch
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from benchmark.downstream.example.model import Model
from benchmark.downstream.example.dataset import RandomDataset


class DownstreamExpert(nn.Module):
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log
    """

    def __init__(self, upstream_dim, downstream='phone', datarc={}, modelrc={}, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.downstream = downstream
        self.datarc = datarc
        self.modelrc = modelrc

        self.train_dataset = RandomDataset(**datarc)
        self.dev_dataset = RandomDataset(**datarc)
        self.test_dataset = RandomDataset(**datarc)

        self.connector = nn.Linear(upstream_dim, modelrc['input_dim'])
        self.model = Model(
            **modelrc, output_class_num=self.train_dataset.class_num)
        self.objective = nn.CrossEntropyLoss()

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
    def forward(self, features, utterance_labels,
                records=None, logger=None, global_step=0):
        """
        Args:
            features:
                list of unpadded features [feat1, feat2, ...]
                each feat is in torch.FloatTensor and already
                put in the device assigned by command-line args

            utterance_labels, ... :
                in the order defined by your dataset and collate_fn
                these are all in cpu, and you can move them to the
                same device as features

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
        features = pad_sequence(features, batch_first=True)
        features = self.connector(features)
        predicted = self.model(features)

        labels = torch.LongTensor(utterance_labels).to(features.device)
        loss = self.objective(predicted, labels)

        predicted_classid = predicted.max(dim=-1).indices
        records['acc'] += (predicted_classid == labels).view(-1).cpu().float().tolist()

        if not self.training:
            # some evaluation-only processing, eg. decoding
            pass

        return loss
