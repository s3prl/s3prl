import os

# import math
import torch

# import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from .model import Model
from .dataset import MOSEIDataset

import pandas as pd

# import sys

from collections import defaultdict
import sklearn

from pathlib import Path


class DownstreamExpert(nn.Module):
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log
    """

    def __init__(self, upstream_dim, downstream_expert, expdir, **kwargs):
        """
        Args:
            upstream_dim: int
                Different upstream will give different representation dimension
                You might want to first project them to the same dimension

            downstream_expert: dict
                The 'downstream_expert' field specified in your downstream config file
                eg. downstream/downstream/example/config.yaml

            **kwargs: dict
                The arguments specified by the argparser in run_downstream.py
                in case you need it.
        """

        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.datarc = downstream_expert["datarc"]
        self.modelrc = downstream_expert["modelrc"]

        # Convert Label File to (filename, label) for each split
        self.train_data, self.dev_data, self.test_data = [], [], []
        df = pd.read_csv(self.datarc["label_path"], encoding="latin-1")
        for row in df.itertuples():
            filename = row.file + "_" + str(row.index) + ".wav"
            # for 2-clas sentiment with negative and non-negative
            # neutral is counted as positive
            if self.datarc["num_class"] == 2:
                label = row.label2a
            # for three-class sentiment with positive, neutral and negative
            elif self.datarc["num_class"] == 3:
                label = (row.label2b + 1) # avoid cuda error: device-side ...
            # for six-class emotion
            elif self.datarc["num_class"] == 6:
                label = row.label6
            # for seven-class sentiment
            elif self.datarc["num_class"] == 7:
                # Avoid CUDA error:
                # device-side assert triggered (due to negative label)
                label = (row.label7 + 3)
            else:
                raise ValueError("Unsupported num_class")
            # split train, dev, test
            if row.split == 0:
                self.train_data.append((filename, label))
            elif row.split == 1:
                self.dev_data.append((filename, label))
            elif row.split == 2:
                self.test_data.append((filename, label))

        self.train_dataset = MOSEIDataset("train",
            self.train_data, self.datarc["data_dir"])
        self.dev_dataset = MOSEIDataset("dev",
            self.dev_data, self.datarc["data_dir"])
        self.test_dataset = MOSEIDataset("test", 
            self.test_data, self.datarc["data_dir"])

        self.connector = nn.Linear(upstream_dim, self.modelrc["input_dim"])
        self.model = Model(output_class_num=self.datarc["num_class"],
            **self.modelrc)
        self.objective = nn.CrossEntropyLoss()
        self.expdir = expdir
        self.logging = os.path.join(self.expdir, "log.log")
        self.best = defaultdict(lambda: 0)
        self.answer = []

    def _get_train_dataloader(self, dataset, epoch: int):
        from s3prl.utility.data import get_ddp_sampler
        sampler = get_ddp_sampler(dataset, epoch)
        return DataLoader(
            dataset, batch_size=self.datarc['train_batch_size'],
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=self.datarc['num_workers'],
            collate_fn=dataset.collate_fn
        )

    def _get_eval_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.datarc["eval_batch_size"],
            shuffle=False,
            num_workers=self.datarc["num_workers"],
            collate_fn=dataset.collate_fn,
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
    def get_train_dataloader(self, epoch: int):
        return self._get_train_dataloader(self.train_dataset, epoch)

    # Interface
    def get_dev_dataloader(self):
        return self._get_eval_dataloader(self.dev_dataset)

    # Interface
    def get_test_dataloader(self):
        return self._get_eval_dataloader(self.test_dataset)

    def get_dataloader(self, mode, epoch: int=0):
        if mode == 'train':
            return eval(f'self.get_{mode}_dataloader')(epoch)
        return eval(f'self.get_{mode}_dataloader')()

    # Interface
    def forward(self, mode, features, labels, records, **kwargs):
        """
        This function will be used in both train/dev/test, you can use
        self.training (bool) to control the different behavior for
        training or evaluation (dev/test)

        Args:
            mode: str
                'train' or 'dev' or 'test'

            features:
                list of unpadded features [feat1, feat2, ...]
                each feat is in torch.FloatTensor and already
                put in the device assigned by command-line args

            records:
                defaultdict(list), by dumping contents into records,
                these contents can be averaged and logged on Tensorboard
                later by self.log_records

                Note1. downstream/runner.py will call self.log_records
                    1. every log_step during training
                    2. once after evalute the whole dev/test dataloader

                Note2. log_step is defined in your downstream config

        Return:
            loss:
                the loss to be optimized, should not be detached
                a single scalar in torch.FloatTensor
        """
        features = pad_sequence(features, batch_first=True)
        features = self.connector(features)
        predicted = self.model(features)

        utterance_labels = labels
        labels = torch.LongTensor(utterance_labels).to(features.device)
        loss = self.objective(predicted, labels)
        predicted_classid = predicted.max(dim=-1).indices

        records["acc"] += (
            predicted_classid == labels).view(-1).cpu().float().tolist()
        # records['filename'] += filenames
        records["predicted"] += predicted_classid.cpu().float().tolist()
        records["original"] += labels.cpu().float().tolist()

        if not self.training:
            # some evaluation-only processing, eg. decoding
            pass

        return loss

    # interface
    def log_records(self, mode, records, logger, global_step, **kwargs):
        """
        This function will be used in both train/dev/test, you can use
        self.training (bool) to control the different behavior for
        training or evaluation (dev/test)

        Args:
            mode: str
                'train' or 'dev' or 'test'

            records:
                defaultdict(list), contents already prepared by self.forward

            logger:
                Tensorboard SummaryWriter
                please use f'{prefix}your_content_name' as key name
                to log your customized contents

            global_step:
                global_step in runner, which is helpful for Tensorboard logging
        """
        prefix = f"mosei/{mode}-"

        average = torch.FloatTensor(records["acc"]).mean().item()
        f1 = sklearn.metrics.f1_score(
            records["original"], records["predicted"], average="macro"
        )

        logger.add_scalar(f"{prefix}acc", average, global_step=global_step)
        if mode in ["dev", "test"]:
            print(f"{prefix}acc: {average}")
        message = f"{mode} at step {global_step}: {average} (acc), {f1} (f1)\n"
        save_ckpt = []

        if average > self.best[prefix]:
            self.best[prefix] = average
            message = f"New best on {message}"
            name = prefix.split("/")[-1].split("-")[0]
            save_ckpt.append(f"{name}-best.ckpt")

            # only saves the prediction from the best model, not the latest
            if mode in ["dev", "test"]:
                with open(Path(self.expdir) / f"{mode}_predict.txt",
                          "w") as file:
                    line = [f"{f} \n" for f in records["predicted"]]
                    file.writelines(line)

        # write true label once
        if mode in ["dev", "test"]:
            with open(Path(self.expdir) / f"{mode}_truth.txt", "w") as file:
                line = [f"{f} \n" for f in records["original"]]
                file.writelines(line)

        with open(self.logging, "a") as f:
            f.write(message)

        if not self.training:
            # some evaluation-only processing, eg. decoding
            pass

        return save_ckpt
