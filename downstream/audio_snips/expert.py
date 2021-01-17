import os
import math
import torch
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.nn.utils.rnn import pad_sequence

from .model import Model
from .dataset import AudioSLUDataset
import pandas as pd
from collections import Counter
import IPython


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
                eg. downstream/example/config.yaml

            **kwargs: dict
                The arguments specified by the argparser in run_downstream.py
                in case you need it.
        """

        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.datarc = downstream_expert['datarc']
        self.modelrc = downstream_expert['modelrc']
        
        self.get_dataset()

        self.train_dataset = []
        self.dev_dataset = []
        self.test_dataset = []

        for speaker_name in self.datarc['train_speakers']:
            self.train_dataset.append(AudioSLUDataset(self.train_df, self.base_path, self.Sy_intent, speaker_name))
            self.dev_dataset.append(AudioSLUDataset(self.valid_df, self.base_path, self.Sy_intent, speaker_name))

        for speaker_name in self.datarc['test_speakers']:
            self.test_dataset.append(AudioSLUDataset(self.test_df, self.base_path, self.Sy_intent, speaker_name))

        self.collate_fn = self.train_dataset[0].collate_fn
        

        # speaker_name = 'Aditi'

        # self.train_dataset = AudioSLUDataset(self.train_df, self.base_path, self.Sy_intent, speaker_name)
        # self.dev_dataset = AudioSLUDataset(self.valid_df, self.base_path, self.Sy_intent, speaker_name)
        # self.test_dataset = AudioSLUDataset(self.test_df, self.base_path, self.Sy_intent, speaker_name)

        self.connector = nn.Linear(upstream_dim, self.modelrc['input_dim'])
        self.model = Model(input_dim=self.modelrc['input_dim'], agg_module=self.modelrc['agg_module'],output_dim=sum(self.values_per_slot), config=self.modelrc)
        self.objective = nn.CrossEntropyLoss()

    def get_dataset(self):
        self.base_path = self.datarc['file_path']
        train_df = pd.read_csv(os.path.join(self.base_path, "data/nlu_annotation", "train"), sep='\t')
        valid_df = pd.read_csv(os.path.join(self.base_path, "data/nlu_annotation", "valid"), sep='\t')
        test_df = pd.read_csv(os.path.join(self.base_path, "data/nlu_annotation", "test"), sep='\t')

        #IPython.embed()
        Sy_intent = {"intent": {}}

        values_per_slot = []
        for slot in ['intent']:
            slot_values = Counter(train_df[slot])
            for index, value in enumerate(slot_values):
                Sy_intent[slot][value] = index
            values_per_slot.append(len(slot_values))
        self.values_per_slot = values_per_slot
        self.Sy_intent = Sy_intent
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df

    def _get_train_dataloader(self, dataset):
        return DataLoader(
            dataset, batch_size=self.datarc['train_batch_size'],
            shuffle=True, num_workers=self.datarc['num_workers'],
            collate_fn=self.collate_fn
        )

    def _get_eval_dataloader(self, dataset):
        return DataLoader(
            dataset, batch_size=self.datarc['eval_batch_size'],
            shuffle=False, num_workers=self.datarc['num_workers'],
            collate_fn=self.collate_fn
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
        return self._get_train_dataloader(ConcatDataset(self.train_dataset))

    # Interface
    def get_dev_dataloader(self):
        return self._get_eval_dataloader(ConcatDataset(self.dev_dataset))

    # Interface
    def get_test_dataloader(self):
        return self._get_eval_dataloader(ConcatDataset(self.test_dataset))

    # Interface
    def forward(self, features, labels,
                records=None, logger=None, prefix=None, global_step=0, **kwargs):
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
        features_pad = pad_sequence(features, batch_first=True)
        
        attention_mask = [torch.ones((feature.shape[0])) for feature in features] 

        attention_mask_pad = pad_sequence(attention_mask,batch_first=True)

        attention_mask_pad = (1.0 - attention_mask_pad) * -100000.0

        features_pad = self.connector(features_pad)
        intent_logits = self.model(features_pad, attention_mask_pad.cuda())

        intent_loss = 0
        start_index = 0
        predicted_intent = []
        #labels = torch.LongTensor(labels).to(features_pad.device)
        
        labels = torch.stack(labels).to(features_pad.device)
        for slot in range(len(self.values_per_slot)):
            end_index = start_index + self.values_per_slot[slot]
            subset = intent_logits[:, start_index:end_index]

            intent_loss += self.objective(subset, labels[:, slot])
            predicted_intent.append(subset.max(1)[1])
            start_index = end_index

        predicted_intent = torch.stack(predicted_intent, dim=1)
        #IPython.embed()
        records['acc'] += (predicted_intent == labels).prod(1).view(-1).cpu().float().tolist()

        if not self.training:
            # some evaluation-only processing, eg. decoding
            #IPython.embed()
            pass

        return intent_loss

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
