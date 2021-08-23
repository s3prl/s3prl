import os
import math
import torch
import random
import pandas as pd
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed import is_initialized
from torch.nn.utils.rnn import pad_sequence

from ..model import *
from .dataset import FluentCommandsDataset


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

        self.get_dataset()

        self.train_dataset = FluentCommandsDataset(self.train_df, self.base_path, self.Sy_intent)
        self.dev_dataset = FluentCommandsDataset(self.valid_df, self.base_path, self.Sy_intent)
        self.test_dataset = FluentCommandsDataset(self.test_df, self.base_path, self.Sy_intent)

        model_cls = eval(self.modelrc['select'])
        model_conf = self.modelrc.get(self.modelrc['select'], {})
        self.projector = nn.Linear(upstream_dim, self.modelrc['projector_dim'])
        self.model = model_cls(
            input_dim = self.modelrc['projector_dim'],
            output_dim = sum(self.values_per_slot),
            **model_conf,
        )
        self.objective = nn.CrossEntropyLoss()
        self.expdir = expdir
        self.register_buffer('best_score', torch.zeros(1))

    def get_dataset(self):
        self.base_path = self.datarc['file_path']
        train_df = pd.read_csv(os.path.join(self.base_path, "data", "train_data.csv"))
        valid_df = pd.read_csv(os.path.join(self.base_path, "data", "valid_data.csv"))
        test_df = pd.read_csv(os.path.join(self.base_path, "data", "test_data.csv"))

        Sy_intent = {"action": {}, "object": {}, "location": {}}

        values_per_slot = []
        for slot in ["action", "object", "location"]:
            slot_values = Counter(train_df[slot])
            for index, value in enumerate(slot_values):
                Sy_intent[slot][value] = index
                Sy_intent[slot][index] = value
            values_per_slot.append(len(slot_values))
        self.values_per_slot = values_per_slot
        self.Sy_intent = Sy_intent
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df

    def _get_train_dataloader(self, dataset):
        sampler = DistributedSampler(dataset) if is_initialized() else None
        return DataLoader(
            dataset, batch_size=self.datarc['train_batch_size'],
            shuffle=(sampler is None), sampler=sampler,
            num_workers=self.datarc['num_workers'],
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
    def forward(self, mode, features, labels, filenames, records, **kwargs):
        labels = [torch.LongTensor(label) for label in labels]
        features_len = torch.IntTensor([len(feat) for feat in features]).to(device=features[0].device)
        features = pad_sequence(features, batch_first=True)

        features = self.projector(features)
        intent_logits, _ = self.model(features, features_len)

        intent_loss = 0
        start_index = 0
        predicted_intent = []
        
        labels = torch.stack(labels).to(features.device)
        for slot in range(len(self.values_per_slot)):
            end_index = start_index + self.values_per_slot[slot]
            subset = intent_logits[:, start_index:end_index]

            intent_loss += self.objective(subset, labels[:, slot])
            predicted_intent.append(subset.max(1)[1])
            start_index = end_index

        predicted_intent = torch.stack(predicted_intent, dim=1)
        records['acc'] += (predicted_intent == labels).prod(1).view(-1).cpu().float().tolist()
        records['intent_loss'].append(intent_loss.item())

        def idx2slots(indices: torch.Tensor):
            action_idx, object_idx, location_idx = indices.cpu().tolist()
            return (
                self.Sy_intent["action"][action_idx],
                self.Sy_intent["object"][object_idx],
                self.Sy_intent["location"][location_idx],
            )

        records["filename"] += filenames
        records["predict"] += list(map(idx2slots, predicted_intent))
        records["truth"] += list(map(idx2slots, labels))

        return intent_loss

    # interface
    def log_records(self, mode, records, logger, global_step, **kwargs):
        save_names = []
        for key in ["acc", "intent_loss"]:
            values = records[key]
            average = torch.FloatTensor(values).mean().item()
            logger.add_scalar(
                f'fluent_commands/{mode}-{key}',
                average,
                global_step=global_step
            )
            with open(Path(self.expdir) / "log.log", 'a') as f:
                if key == 'acc':
                    print(f"{mode} {key}: {average}")
                    f.write(f'{mode} at step {global_step}: {average}\n')
                    if mode == 'dev' and average > self.best_score:
                        self.best_score = torch.ones(1) * average
                        f.write(f'New best on {mode} at step {global_step}: {average}\n')
                        save_names.append(f'{mode}-best.ckpt')

        with open(Path(self.expdir) / f"{mode}_predict.csv", "w") as file:
            lines = [f"{f},{a},{o},{l}\n" for f, (a, o, l) in zip(records["filename"], records["predict"])]
            file.writelines(lines)

        with open(Path(self.expdir) / f"{mode}_truth.csv", "w") as file:
            lines = [f"{f},{a},{o},{l}\n" for f, (a, o, l) in zip(records["filename"], records["truth"])]
            file.writelines(lines)

        return save_names
