"""Downstream expert for Spoken Term Detection on Speech Commands."""

import re
import hashlib
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence

from .model import Model
from .dataset import SpeechCommandsDataset, SpeechCommandsTestingDataset


class DownstreamExpert(nn.Module):
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log
    """

    def __init__(self, upstream_dim: int, downstream_expert: dict, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.datarc = downstream_expert["datarc"]
        self.modelrc = downstream_expert["modelrc"]

        train_list, valid_list = split_dataset(self.datarc["speech_commands_root"])

        self.train_dataset = SpeechCommandsDataset(train_list, **self.datarc)
        self.dev_dataset = SpeechCommandsDataset(valid_list, **self.datarc)
        self.test_dataset = SpeechCommandsTestingDataset(**self.datarc)

        self.model = Model(
            input_dim=upstream_dim,
            output_class_num=self.train_dataset.class_num,
            **self.modelrc,
        )
        self.objective = nn.CrossEntropyLoss()

    def _get_balanced_dataloader(self, dataset, drop_last=False):
        return DataLoader(
            dataset,
            sampler=WeightedRandomSampler(
                dataset.sample_weights, len(dataset.sample_weights)
            ),
            batch_size=self.datarc["batch_size"],
            drop_last=drop_last,
            num_workers=self.datarc["num_workers"],
            collate_fn=dataset.collate_fn,
        )

    def _get_dataloader(self, dataset):
        return DataLoader(
            dataset,
            shuffle=False,
            batch_size=self.datarc["batch_size"],
            drop_last=False,
            num_workers=self.datarc["num_workers"],
            collate_fn=dataset.collate_fn,
        )

    # Interface
    def get_train_dataloader(self):
        return self._get_balanced_dataloader(self.train_dataset, drop_last=True)

    # Interface
    def get_dev_dataloader(self):
        return self._get_balanced_dataloader(self.dev_dataset, drop_last=False)

    # Interface
    def get_test_dataloader(self):
        return self._get_dataloader(self.test_dataset)

    # Interface
    def forward(self, features, labels, records, logger, prefix, global_step, **kwargs):
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

                Note1. benchmark/runner.py will call self.log_records
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
        features = pad_sequence(features, batch_first=True)
        predicted = self.model(features)

        labels = torch.LongTensor(labels).to(features.device)
        loss = self.objective(predicted, labels)

        predicted_classid = predicted.max(dim=-1).indices
        records["acc"] += (predicted_classid == labels).view(-1).cpu().float().tolist()

        return loss

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
            logger.add_scalar(f"{prefix}{key}", average, global_step=global_step)


def split_dataset(
    speech_commands_root, max_uttr_per_class=2 ** 27 - 1
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Split all speech commands into 3 set.
    
    Return:
        train_list: [(class, audio_path), ...]
        valid_list: as above
    """
    data_root_path = Path(speech_commands_root)

    all_classes = [
        entry.name
        for entry in data_root_path.iterdir()
        if entry.is_dir() and entry.name != "_background_noise_"
    ]

    train_list, valid_list = [], []

    for class_name in all_classes:
        for audio_path in (data_root_path / class_name).glob("*.wav"):
            speaker_hashed = re.sub(r"_nohash_.*$", "", audio_path.name)
            hashed_again = hashlib.sha1(speaker_hashed.encode("utf-8")).hexdigest()
            percentage_hash = (int(hashed_again, 16) % (max_uttr_per_class + 1)) * (
                100.0 / max_uttr_per_class
            )
            if percentage_hash < 10:
                valid_list.append((class_name, audio_path))
            elif percentage_hash < 20:
                pass  # testing set is discarded
            else:
                train_list.append((class_name, audio_path))

    return train_list, valid_list
