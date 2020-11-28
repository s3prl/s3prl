import os
import math
import torch
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from benchmark.downstream.phone_linear.model import Model
from benchmark.downstream.phone_linear.dataset import PhoneDataset


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

    def __init__(self, upstream_dim, downstream='phone_linear', datarc={}, modelrc={}, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.downstream = downstream
        self.datarc = datarc
        self.modelrc = modelrc

        self.train_dataset = PhoneDataset('train', self.datarc['train_batch_size'], **datarc)
        self.dev_dataset = PhoneDataset('dev', self.datarc['eval_batch_size'], **datarc)
        self.test_dataset = PhoneDataset('test', self.datarc['eval_batch_size'], **datarc)

        self.model = Model(input_dim=self.upstream_dim, output_class_num=self.train_dataset.class_num)
        self.objective = nn.CrossEntropyLoss()

    def _get_train_dataloader(self, dataset):
        return DataLoader(
            dataset, batch_size=1, # for bucketing
            shuffle=True, num_workers=self.datarc['num_workers'],
            drop_last=False, pin_memory=True
        )

    def _get_eval_dataloader(self, dataset):
        return DataLoader(
            dataset, batch_size=1, # for bucketing
            shuffle=False, num_workers=self.datarc['num_workers'],
            drop_last=False, pin_memory=True
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

    def _match_length(self, inputs, labels):
        """
        Since the upstream extraction process can sometimes cause a mismatch
        between the seq lenth of inputs and labels:
        - if len(inputs) > len(labels), we truncate the final few timestamp of inputs to match the length of labels
        - if len(inputs) < len(labels), we duplicate the last timestep of inputs to match the length of labels
        Note that the length of labels should never be changed.
        """
        input_len, label_len = inputs.size(1), labels.size(-1)
        if input_len > label_len:
            inputs = inputs[:, :label_len, :]
        elif input_len < label_len:
            pad_vec = inputs[:, -1, :].unsqueeze(1) # (batch_size, 1, feature_dim)
            inputs = torch.cat((inputs, pad_vec.repeat(1, label_len-input_len, 1)), dim=1) # (batch_size, seq_len, feature_dim), where seq_len == labels.size(-1)
        return inputs, labels

    # Interface
    def forward(self, features, labels,
                records=None, logger=None, global_step=0):
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
        if labels.size(0) == 1: labels = labels.squeeze(0) # hack bucketing
        labels = torch.LongTensor(labels).to(features.device)

        
        predicted = self.model(features)
        predicted, labels = self._match_length(predicted, labels) # match the predicted instead of features

        # cause logits are in (batch, seq, class) and labels are in (batch, seq)
        # nn.CrossEntropyLoss expect to have (N, class) and (N,) as input
        # here we flatten logits and labels in order to apply nn.CrossEntropyLoss
        class_num = predicted.size(-1)
        loss = self.objective(predicted.reshape(-1, class_num), labels.reshape(-1))

        predicted_classid = predicted.max(dim=-1).indices
        records['acc'] += (predicted_classid == labels).view(-1).cpu().float().tolist()

        return loss
