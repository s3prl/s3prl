import os
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from torch.distributed import is_initialized
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from .dataset import VCC16SegmentalSystemLevelDataset, VCC18SegmentalDataset
from .model import Model

warnings.filterwarnings("ignore")


class DownstreamExpert(nn.Module):
    def __init__(self, upstream_dim, downstream_expert, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.datarc = downstream_expert["datarc"]
        self.modelrc = downstream_expert["modelrc"]

        self.train_dataset = VCC18SegmentalDataset(
            preprocess(self.datarc["file_path"], "train"), self.datarc["file_path"]
        )
        self.dev_dataset = VCC18SegmentalDataset(
            preprocess(self.datarc["file_path"], "valid"), self.datarc["file_path"]
        )
        self.test_dataset = VCC18SegmentalDataset(
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
        self.objective = nn.MSELoss()

        self.best_scores = {
            "dev": -np.inf,
            "test": -np.inf,
        }

        self.system_level_dataset = VCC16SegmentalSystemLevelDataset(
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
    def forward(self, mode, features, prefix_sums, system_names, records, **kwargs):

        features = torch.stack(features)
        features = self.connector(features)
        segments_scores = self.model(features)

        uttr_scores = []
        for i in range(len(prefix_sums) - 1):
            current_segment_scores = segments_scores[
                prefix_sums[i] : prefix_sums[i + 1]
            ]
            uttr_score = current_segment_scores.mean(dim=-1)
            uttr_scores.append(uttr_score.detach().cpu())

        if mode == "test_system":
            if len(records["system"]) == 0:
                records["system"].append(defaultdict(list))
            for i in range(len(system_names)):
                records["system"][0][system_names[i]].append(uttr_scores[i].tolist())

        return 0

    # interface
    def log_records(
        self, mode, records, logger, global_step, batch_ids, total_batch_num, **kwargs
    ):
        if mode == "test_system":
            all_system_pred_scores = []
            all_system_true_scores = []

            for key, values in records["system"][0].items():
                all_system_pred_scores.append(np.mean(values))
                all_system_true_scores.append(self.system_level_mos[key].iloc[0])

            all_system_pred_scores = np.array(all_system_pred_scores)
            all_system_true_scores = np.array(all_system_true_scores)

            MSE = np.mean((all_system_true_scores - all_system_pred_scores) ** 2)
            pearson_rho, _ = pearsonr(all_system_true_scores, all_system_pred_scores)
            spearman_rho, _ = spearmanr(all_system_true_scores, all_system_pred_scores)

            tqdm.write(f"[{mode}] System Level MSE  = {MSE:.4f}")
            tqdm.write(f"[{mode}] System Level LCC  = {pearson_rho:.4f}")
            tqdm.write(f"[{mode}] System Level SRCC = {spearman_rho:.4f}")


def preprocess(base_path, txt_file):
    dataframe = pd.read_csv(Path(base_path, txt_file), index_col=False)
    return dataframe