import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import torch
import torch.nn as nn
from torch.distributed import is_initialized
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from .dataset import VCC18Dataset, VCC16Dataset
from .model import Model

warnings.filterwarnings("ignore")


class DownstreamExpert(nn.Module):
    def __init__(self, upstream_dim, downstream_expert, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.datarc = downstream_expert["datarc"]
        self.modelrc = downstream_expert["modelrc"]

        self.train_dataset = VCC18Dataset(
            preprocess(self.datarc["vcc2018_file_path"], "train"),
            self.datarc["vcc2018_file_path"],
        )
        self.dev_dataset = VCC18Dataset(
            preprocess(self.datarc["vcc2018_file_path"], "valid"),
            self.datarc["vcc2018_file_path"],
        )
        self.vcc2018_test_dataset = VCC18Dataset(
            preprocess(self.datarc["vcc2018_file_path"], "test"),
            self.datarc["vcc2018_file_path"],
        )

        self.vcc2018_system_mos = pd.read_csv(
            Path(
                self.datarc["vcc2018_file_path"],
                "VCC2018_Results/system_mos_all_trackwise.csv",
            )
        )

        self.vcc2016_test_dataset = VCC16Dataset(
            list(
                Path.iterdir(Path(self.datarc["vcc2016_file_path"], "unified_speech"))
            ),
            Path(self.datarc["vcc2016_file_path"], "unified_speech"),
        )

        self.vcc2016_system_mos = pd.read_csv(
            Path(self.datarc["vcc2016_file_path"], "system_mos.csv"), index_col=False
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

        self.best_scores = {
            "dev_loss": np.inf,
            "dev_LCC": -np.inf,
            "dev_SRCC": -np.inf,
            "vcc2016_test_LCC": -np.inf,
            "vcc2016_test_SRCC": -np.inf,
        }

    # Interface
    def get_dataloader(self, mode):
        if mode == "train":
            return self._get_train_dataloader(self.train_dataset)
        elif mode == "dev":
            return self._get_eval_dataloader(self.dev_dataset)
        elif mode == "vcc2018_test":
            return self._get_eval_dataloader(self.vcc2018_test_dataset)
        elif mode == "vcc2016_test":
            return self._get_eval_dataloader(self.vcc2016_test_dataset)

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
    def forward(self, mode, features, scores, system_names, records, **kwargs):
        device = features[0].device

        lengths = torch.Tensor([len(feat) for feat in features]).to(device)

        features = pad_sequence(features, batch_first=True)
        features = self.connector(features)
        frame_scores, uttr_scores = self.model(features, lengths)

        if mode == "train" or mode == "dev" or mode == "vcc2018_test":
            scores = scores.to(device)
            frame_loss = self.objective(frame_scores, scores.expand_as(frame_scores))
            uttr_loss = self.objective(uttr_scores, scores).mean()

            mask = (
                torch.arange(frame_loss.size(-1)).expand_as(frame_loss).to(device)
                < lengths[:, None]
            ).float()
            frame_loss = (frame_loss * mask).sum(dim=-1).div(lengths).mean()
            loss = frame_loss + uttr_loss

            records["total loss"].append(loss.item())

            records["pred_scores"] += uttr_scores.detach().view(-1).cpu().tolist()
            records["true_scores"] += scores.detach().cpu().tolist()

        if mode == "vcc2016_test":
            records["pred_scores"] += uttr_scores.detach().view(-1).cpu().tolist()

        if len(records["system"]) == 0:
            records["system"].append(defaultdict(list))
        for i in range(len(system_names)):
            records["system"][0][system_names[i]].append(uttr_scores[i].tolist())

        if mode == "train":
            records["frame loss"].append(frame_loss.item())
            records["utterance loss"].append(uttr_loss.item())
            return loss

        return 0

    # interface
    def log_records(
        self, mode, records, logger, global_step, batch_ids, total_batch_num, **kwargs
    ):
        save_names = []

        # logging loss

        if mode == "train":
            avg_uttr_loss = torch.FloatTensor(records["utterance loss"]).mean().item()
            avg_frame_loss = torch.FloatTensor(records["frame loss"]).mean().item()

            logger.add_scalar(
                f"wav2MOS_linear/{mode}-utterance loss",
                avg_uttr_loss,
                global_step=global_step,
            )
            logger.add_scalar(
                f"wav2MOS_linaer/{mode}-frame loss",
                avg_frame_loss,
                global_step=global_step,
            )

        if mode == "train" or mode == "dev":
            avg_total_loss = torch.FloatTensor(records["total loss"]).mean().item()

            logger.add_scalar(
                f"wav2MOS_linear/{mode}-total loss",
                avg_total_loss,
                global_step=global_step,
            )

        # logging Utterance-level MSE, LCC, SRCC
        # import IPython

        # IPython.embed()
        if mode == "dev" or mode == "vcc2018_test":
            # some evaluation-only processing, eg. decoding
            all_pred_scores = records["pred_scores"]
            all_true_scores = records["true_scores"]
            all_pred_scores = np.array(all_pred_scores)
            all_true_scores = np.array(all_true_scores).reshape(-1)
            MSE = np.mean((all_true_scores - all_pred_scores) ** 2)
            logger.add_scalar(
                f"wav2MOS_linear/{mode}-Utterance level MSE",
                MSE,
                global_step=global_step,
            )
            pearson_rho, _ = pearsonr(all_true_scores, all_pred_scores)
            logger.add_scalar(
                f"wav2MOS_linear/{mode}-Utterance level LCC",
                pearson_rho,
                global_step=global_step,
            )
            spearman_rho, _ = spearmanr(all_true_scores.T, all_pred_scores.T)
            logger.add_scalar(
                f"wav2MOS_linear/{mode}-Utterance level SRCC",
                spearman_rho,
                global_step=global_step,
            )

            tqdm.write(f"[{mode}] Utterance-level MSE  = {MSE:.4f}")
            tqdm.write(f"[{mode}] Utterance-level LCC  = {pearson_rho:.4f}")
            tqdm.write(f"[{mode}] Utterance-level SRCC = {spearman_rho:.4f}")

        # select which system mos to use
        if mode == "dev" or mode == "vcc2018_test":
            system_level_mos = self.vcc2018_system_mos

        if mode == "vcc2016_test":
            system_level_mos = self.vcc2016_system_mos

        # logging System-level MSE, LCC, SRCC
        if mode == "dev" or mode == "vcc2018_test" or mode == "vcc2016_test":
            all_system_pred_scores = []
            all_system_true_scores = []

            for key, values in records["system"][0].items():
                all_system_pred_scores.append(np.mean(values))
                all_system_true_scores.append(system_level_mos[key].iloc[0])

            all_system_pred_scores = np.array(all_system_pred_scores)
            all_system_true_scores = np.array(all_system_true_scores).reshape(-1)

            MSE = np.mean((all_system_true_scores - all_system_pred_scores) ** 2)
            pearson_rho, _ = pearsonr(all_system_true_scores, all_system_pred_scores)
            spearman_rho, _ = spearmanr(all_system_true_scores, all_system_pred_scores)

            tqdm.write(f"[{mode}] System-level MSE  = {MSE:.4f}")
            tqdm.write(f"[{mode}] System-level LCC  = {pearson_rho:.4f}")
            tqdm.write(f"[{mode}] System-level SRCC = {spearman_rho:.4f}")

            logger.add_scalar(
                f"wav2MOS_linear/{mode}-System level MSE",
                MSE,
                global_step=global_step,
            )
            logger.add_scalar(
                f"wav2MOS_linear/{mode}-System level LCC",
                pearson_rho,
                global_step=global_step,
            )
            logger.add_scalar(
                f"wav2MOS_linear/{mode}-System level SRCC",
                spearman_rho,
                global_step=global_step,
            )

        # save model
        if mode == "dev":
            if avg_total_loss < self.best_scores["dev_loss"]:
                self.best_scores[mode] = avg_total_loss
                save_names.append(f"{mode}-best.ckpt")
            if pearson_rho > self.best_scores["dev_LCC"]:
                self.best_scores["dev_LCC"] = pearson_rho
                save_names.append(f"{mode}-LCC-best.ckpt")
            if spearman_rho > self.best_scores["dev_SRCC"]:
                self.best_scores["dev_SRCC"] = spearman_rho
                save_names.append(f"{mode}-SRCC-best.ckpt")
        if mode == "vcc2016_test":
            if pearson_rho > self.best_scores["vcc2016_test_LCC"]:
                self.best_scores["vcc2016_test_LCC"] = pearson_rho
                save_names.append(f"{mode}-LCC-best.ckpt")
            if spearman_rho > self.best_scores["vcc2016_test_SRCC"]:
                self.best_scores["vcc2016_test_SRCC"] = spearman_rho
                save_names.append(f"{mode}-SRCC-best.ckpt")

        return save_names


def preprocess(base_path, txt_file):
    dataframe = pd.read_csv(Path(base_path, txt_file), index_col=False)
    return dataframe