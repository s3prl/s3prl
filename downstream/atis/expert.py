import os
import math
import torch
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from .model import Model
from .dataset import AtisDataset
import pandas as pd
from collections import Counter


class DownstreamExpert(nn.Module):
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log
    """

    def __init__(self, upstream_dim, downstream_expert, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.datarc = downstream_expert['datarc']
        self.modelrc = downstream_expert['modelrc']
        
        self.get_dataset()

        self.train_dataset = AtisDataset(self.train_df, self.base_path, self.Sy_intent, 'train')
        self.dev_dataset = AtisDataset(self.valid_df, self.base_path, self.Sy_intent, 'dev')
        self.test_dataset = AtisDataset(self.test_df, self.base_path, self.Sy_intent, 'test')

        self.connector = nn.Linear(upstream_dim, self.modelrc['input_dim'])
        self.model = Model(input_dim=self.modelrc['input_dim'], agg_module=self.modelrc['agg_module'],output_dim=sum(self.values_per_slot), config=self.modelrc)
        self.objective = nn.CrossEntropyLoss()

    def get_dataset(self):
        self.base_path = self.datarc['file_path']
        train_df = pd.read_csv(os.path.join(self.base_path, "nlu_iob", "iob.train"), sep='\t', header=None)
        valid_df = pd.read_csv(os.path.join(self.base_path, "nlu_iob", "iob.dev"), sep='\t', header=None)
        test_df = pd.read_csv(os.path.join(self.base_path, "nlu_iob", "iob.test"), sep='\t', header=None)

        train_dict = {"id": {}, "intent": {}}
        valid_dict = {"id": {}, "intent": {}}
        test_dict = {"id": {}, "intent": {}}

        
        for dc, df, type_name in [(train_dict, train_df, 'train'), (valid_dict, valid_df, 'dev'), (test_dict, test_df, 'test')]:
            n = 0
            for i in range(len(df)):
                if(os.path.exists( os.path.join(self.base_path, type_name, df[0][i].split()[0]+'.wav')) & (df[0][i].split()[0] not in dc['id'].values())):
                    dc['id'][n] = df[0][i].split()[0]
                    dc['intent'][n] = df[1][i].split()[-1]
                    n += 1
                # else:
                #     print(os.path.join(self.base_path, type_name, df[0][i].split()[0]+'.wav'))
            print(type_name, ': ', n+1)    

        train_df = pd.DataFrame(data=train_dict)
        valid_df = pd.DataFrame(data=valid_dict)
        test_df = pd.DataFrame(data=test_dict)


        Sy_intent = {"intent": {}}
        values_per_slot = []
        
        for slot in ["intent"]:
            slot_values = Counter(train_df[slot]) + Counter(valid_df[slot]) + Counter(test_df[slot])
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

    def get_dataloader(self, mode):
        return eval(f'self.get_{mode}_dataloader')()

    # Interface
    def forward(self, mode, features, labels, records=None, **kwargs):

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
        records['acc'] += (predicted_intent == labels).prod(1).view(-1).cpu().float().tolist()
        records['intent_loss'].append(intent_loss.item())

        return intent_loss

    # interface
    def log_records(self, mode, records, logger, global_step, **kwargs):
        for key, values in records.items():
            average = torch.FloatTensor(values).mean().item()
            logger.add_scalar(
                f'atis/{mode}-{key}',
                average,
                global_step=global_step
            )
