import math
import os
import random
import warnings

import numpy as np
import pandas as pd
import scipy.stats
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.distributed import is_initialized
from torch.utils.data import DataLoader, DistributedSampler

from .dataset import VCC18Dataset
from .model import Model

warnings.filterwarnings("ignore")


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
        self.datarc = downstream_expert["datarc"]
        self.modelrc = downstream_expert["modelrc"]

        self.train_dataset = VCC18Dataset(
            preprocess(self.datarc["file_path"], "train"), self.datarc["file_path"]
        )
        self.dev_dataset = VCC18Dataset(
            preprocess(self.datarc["file_path"], "valid"), self.datarc["file_path"]
        )
        self.test_dataset = VCC18Dataset(
            preprocess(self.datarc["file_path"], "test"), self.datarc["file_path"]
        )

        # model_cls = eval(self.modelrc["select"])
        # model_conf = self.modelrc.get(self.modelrc["select"], {})

        self.connector = nn.Linear(upstream_dim, self.modelrc["projector_dim"])
        self.model = Model(input_dim=self.modelrc["projector_dim"], **self.modelrc)
        self.objective = nn.MSELoss()

        self.register_buffer("best_score", torch.zeros(1))

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

        if mode == "train":
            return self._get_train_dataloader(self.train_dataset)
        elif mode == "dev":
            return self._get_eval_dataloader(self.dev_dataset)
        elif mode == "test":
            return self._get_eval_dataloader(self.test_dataset)

    def _get_train_dataloader(self, dataset):
        sampler = DistributedSampler(dataset) if is_initialized() else None
        return DataLoader(
            dataset,
            batch_size=self.datarc["train_batch_size"],
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=self.datarc["num_workers"],
            collate_fn=dataset.collate_fn,
        )

    def _get_eval_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.datarc["eval_batch_size"],
            shuffle=False,
            num_workers=self.datarc["num_workers"],
            collate_fn=dataset.collate_fn,
        )

    # Interface
    def forward(self, mode, features, scores, records, **kwargs):
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
        # scores = [torch.LongTensor(score) for score in scores]
        # features_len = torch.IntTensor([len(feat) for feat in features]).to(
        #     device=features[0].device
        # )

        features = pad_sequence(features, batch_first=True)
        features = self.connector(features)
        frame_scores, uttr_scores = self.model(features)

        scores = scores.to(features.device)
        frame_loss = self.objective(frame_scores, scores[:, None])
        uttr_loss = self.objective(uttr_scores, scores)
        loss = frame_loss + uttr_loss

        # predicted_classid = predicted.max(dim=-1).indices
        # records["acc"] += (predicted_classid == labels).view(-1).cpu().float().tolist()

        if mode == "train":
            records["frame loss"].append(frame_loss.item())
            records["utterance loss"].append(uttr_loss.item())
            records["total loss"].append(loss.item())

        if mode == "dev" or mode == "test":
            records["pred_scores"] += uttr_scores.detach().cpu().tolist()
            records["true_scores"] += scores.detach().cpu().tolist()

        return loss

    # interface
    def log_records(
        self, mode, records, logger, global_step, batch_ids, total_batch_num, **kwargs
    ):
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

        save_names = []

        if mode == "train":
            avg_total_loss = torch.FloatTensor(records["total loss"]).mean().item()
            avg_uttr_loss = torch.FloatTensor(records["utterance loss"]).mean().item()
            avg_frame_loss = torch.FloatTensor(records["frame loss"]).mean().item()

            logger.add_scalar(
                f"wav2MOS/{mode}-total loss", avg_total_loss, global_step=global_step
            )
            logger.add_scalar(
                f"wav2MOS/{mode}-utterance loss", avg_uttr_loss, global_step=global_step
            )
            logger.add_scalar(
                f"wav2MOS/{mode}-frame loss", avg_frame_loss, global_step=global_step
            )

        if mode == "dev" or mode == "test":
            # some evaluation-only processing, eg. decoding
            all_pred_scores = records["pred_scores"]
            all_true_scores = records["true_scores"]
            all_pred_scores = np.array(all_pred_scores)
            all_true_scores = np.array(all_true_scores)
            MSE = np.mean((all_true_scores - all_pred_scores) ** 2)
            logger.add_scalar(f"wav2MOS/{mode}-MSE", MSE, global_step=global_step)
            LCC = np.corrcoef(all_true_scores, all_pred_scores)
            logger.add_scalar(f"wav2MOS/{mode}-LCC", LCC[0][1], global_step=global_step)
            SRCC = scipy.stats.spearmanr(all_true_scores.T, all_pred_scores.T)
            logger.add_scalar(f"wav2MOS/{mode}-SRCC", SRCC[0], global_step=global_step)

            if LCC[0][1] > self.best_score:
                self.best_score = torch.ones(1) * LCC[0][1]
                save_names.append(f"{mode}-best.ckpt")

        return save_names


def preprocess(base_path, txt_file):
    dataframe = pd.read_csv(os.path.join(base_path, txt_file), index_col=False)
    return dataframe
