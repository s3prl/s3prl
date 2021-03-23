import warnings
from pathlib import Path
import os

import numpy as np
import pandas as pd
import scipy.stats
import torch
import torch.nn as nn
from torch.distributed import is_initialized
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from collections import defaultdict

from .dataset import VCC18Dataset, VCC16SystemLevelDataset
from .model import Model

warnings.filterwarnings("ignore")


class DownstreamExpert(nn.Module):
    def __init__(self, upstream_dim, downstream_expert, **kwargs):
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

        self.connector = nn.Linear(upstream_dim, self.modelrc["projector_dim"])
        self.model = Model(
            input_dim=self.modelrc["projector_dim"],
            clipping=self.modelrc["clipping"] if "clipping" in self.modelrc else False,
            attention_pooling=self.modelrc["attention_pooling"]
            if "attention_pooling" in self.modelrc
            else False,
        )
        self.objective = nn.MSELoss(reduction="none")

        # self.register_buffer("best_score", torch.zeros(1))

        self.best_scores = {
            "dev": -np.inf,
            "test": -np.inf,
        }

        self.system_level_dataset = VCC16SystemLevelDataset(
            os.listdir("/livingrooms/public/VCC_2016/unified_speech"),
            "/livingrooms/public/VCC_2016/unified_speech",
        )

        self.system_level_mos = pd.read_csv(
            "/livingrooms/public/VCC_2016/system_mos.csv", index_col=False
        )

    # Interface
    def get_dataloader(self, mode):
        if mode == "train":
            return self._get_train_dataloader(self.train_dataset)
        elif mode == "dev":
            return self._get_eval_dataloader(self.dev_dataset)
        elif mode == "test":
            return self._get_eval_dataloader(self.test_dataset)
        elif mode == "test_system":
            return self._get_eval_dataloader(self.system_level_dataset)

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
    def forward(self, mode, features, system_names, records, **kwargs):
        device = features[0].device

        lengths = torch.Tensor([len(feat) for feat in features]).to(device)

        features = pad_sequence(features, batch_first=True)
        features = self.connector(features)
        frame_scores, uttr_scores = self.model(features, lengths)

        if mode == "test_system":
            if len(records["system"]) == 0:
                records["system"].append(defaultdict(list))
            for i in range(len(system_names)):
                records["system"][0][system_names[i]] += (
                    uttr_scores[i].detach().view(-1).cpu().tolist()
                )

        return 0

    # interface
    def log_records(
        self, mode, records, logger, global_step, batch_ids, total_batch_num, **kwargs
    ):
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

            if LCC[0][1] > self.best_scores[mode]:
                self.best_scores[mode] = LCC[0][1]
                save_names.append(f"{mode}-best.ckpt")

            tqdm.write(f"[{mode}] MSE  = {MSE:.4f}")
            tqdm.write(f"[{mode}] LCC  = {LCC[0][1]:.4f}")
            tqdm.write(f"[{mode}] SRCC = {SRCC[0]:.4f}")

        if mode == "test_system":
            # some evaluation-only processing, eg. decoding

            all_system_pred_scores = []
            all_system_true_scores = []
            for key, values in records["system"][0].items():
                all_system_pred_scores.append(np.mean(values))
                all_system_true_scores.append(self.system_level_mos[key].iloc[0])

            all_system_pred_scores = np.array(all_system_pred_scores)
            all_system_true_scores = np.array(all_system_true_scores)

            MSE = np.mean((all_system_true_scores - all_system_pred_scores) ** 2)
            LCC = np.corrcoef(all_system_true_scores, all_system_pred_scores)
            SRCC = scipy.stats.spearmanr(
                all_system_true_scores.T, all_system_pred_scores.T
            )

            tqdm.write(f"[{mode}] System Level MSE  = {MSE:.4f}")
            tqdm.write(f"[{mode}] System Level LCC  = {LCC[0][1]:.4f}")
            tqdm.write(f"[{mode}] System Level SRCC = {SRCC[0]:.4f}")

        return save_names


def preprocess(base_path, txt_file):
    dataframe = pd.read_csv(Path(base_path, txt_file), index_col=False)
    return dataframe
