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

from .dataset import VCC18SegmentalDataset, VCC16SegmentalDataset
from .model import Model

warnings.filterwarnings("ignore")


class DownstreamExpert(nn.Module):
    def __init__(self, upstream_dim, downstream_expert, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.datarc = downstream_expert["datarc"]
        self.modelrc = downstream_expert["modelrc"]
        idtable = Path(kwargs["expdir"]) / "idtable.pkl"

        self.train_dataset = VCC18SegmentalDataset(
            preprocess(self.datarc["vcc2018_file_path"], "train_judge.csv"),
            self.datarc["vcc2018_file_path"],
            idtable=idtable,
        )
        self.dev_dataset = VCC18SegmentalDataset(
            preprocess(self.datarc["vcc2018_file_path"], "valid_judge.csv"),
            self.datarc["vcc2018_file_path"],
            idtable=idtable,
            valid=False,
        )
        self.vcc2018_test_dataset = VCC18SegmentalDataset(
            preprocess(self.datarc["vcc2018_file_path"], "test_judge.csv"),
            self.datarc["vcc2018_file_path"],
            idtable=idtable,
            valid=False,
        )

        self.vcc2018_system_mos = pd.read_csv(
            Path(
                self.datarc["vcc2018_file_path"],
                "VCC2018_Results/system_mos_all_trackwise.csv",
            )
        )

        self.vcc2016_test_dataset = VCC16SegmentalDataset(
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
            num_judges=5000,
        )
        self.objective = nn.MSELoss()
        self.segment_weight = self.modelrc["segment_weight"]
        self.bias_weight = self.modelrc["bias_weight"]

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
    def forward(
        self,
        mode,
        features,
        prefix_sums,
        means,
        system_names,
        moses,
        judge_ids,
        records,
        **kwargs,
    ):

        features = torch.stack(features)
        features = self.connector(features)

        uttr_scores = []
        bias_scores = []
        if mode == "train":
            means = means.to(features.device)
            judge_ids = judge_ids.to(features.device)
            moses = moses.to(features.device)
            segments_scores, segments_bias_scores = self.model(
                features, judge_ids=judge_ids
            )
            segments_loss = 0
            uttr_loss = 0
            bias_loss = 0
            for i in range(len(prefix_sums) - 1):
                current_segment_scores = segments_scores[
                    prefix_sums[i] : prefix_sums[i + 1]
                ]
                current_bias_scores = segments_bias_scores[
                    prefix_sums[i] : prefix_sums[i + 1]
                ]
                uttr_score = current_segment_scores.mean(dim=-1)
                uttr_scores.append(uttr_score.detach().cpu())
                bias_score = current_bias_scores.mean(dim=-1)
                bias_scores.append(bias_score.detach().cpu())
                segments_loss += self.objective(current_segment_scores, means[i])
                uttr_loss += self.objective(uttr_score, means[i])
                bias_loss += self.objective(bias_score, moses[i])
            segments_loss /= len(prefix_sums) - 1
            uttr_loss /= len(prefix_sums) - 1
            bias_loss /= len(prefix_sums) - 1
            loss = (
                self.segment_weight * segments_loss
                + self.bias_weight * bias_loss
                + uttr_loss
            )

            # for i in range(5):
            #     print(uttr_scores[i], bias_scores[i])

            records["segment loss"].append(segments_loss.item())
            records["utterance loss"].append(uttr_loss.item())
            records["bias loss"].append(bias_loss.item())
            records["total loss"].append(loss.item())

            records["pred_scores"] += uttr_scores
            records["true_scores"] += means.detach().cpu().tolist()

        if mode == "dev" or mode == "vcc2018_test":
            means = means.to(features.device)
            segments_scores = self.model(features)
            segments_loss = 0
            uttr_loss = 0
            for i in range(len(prefix_sums) - 1):
                current_segment_scores = segments_scores[
                    prefix_sums[i] : prefix_sums[i + 1]
                ]
                uttr_score = current_segment_scores.mean(dim=-1)
                uttr_scores.append(uttr_score.detach().cpu())
                segments_loss += self.objective(current_segment_scores, means[i])
                uttr_loss += self.objective(uttr_score, means[i])
            segments_loss /= len(prefix_sums) - 1
            uttr_loss /= len(prefix_sums) - 1
            loss = segments_loss + uttr_loss

            records["total loss"].append(loss.item())

            records["pred_scores"] += uttr_scores
            records["true_scores"] += means.detach().cpu().tolist()

        if mode == "vcc2016_test":
            segments_scores = self.model(features)
            for i in range(len(prefix_sums) - 1):
                current_segment_scores = segments_scores[
                    prefix_sums[i] : prefix_sums[i + 1]
                ]
                uttr_score = current_segment_scores.mean(dim=-1)
                uttr_scores.append(uttr_score.detach().cpu())

        if len(records["system"]) == 0:
            records["system"].append(defaultdict(list))
        for i in range(len(system_names)):
            records["system"][0][system_names[i]].append(uttr_scores[i].tolist())

        if mode == "train":
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
            avg_frame_loss = torch.FloatTensor(records["segment loss"]).mean().item()
            avg_bias_loss = torch.FloatTensor(records["bias loss"]).mean().item()

            logger.add_scalar(
                f"wav2MOS_segment_MBNet/{mode}-utterance loss",
                avg_uttr_loss,
                global_step=global_step,
            )
            logger.add_scalar(
                f"wav2MOS_segment_MBNet/{mode}-segment loss",
                avg_frame_loss,
                global_step=global_step,
            )
            logger.add_scalar(
                f"wav2MOS_segment_MBNet/{mode}-bias loss",
                avg_bias_loss,
                global_step=global_step,
            )

        if mode == "train" or mode == "dev":
            avg_total_loss = torch.FloatTensor(records["total loss"]).mean().item()

            logger.add_scalar(
                f"wav2MOS_segment_MBNet/{mode}-total loss",
                avg_total_loss,
                global_step=global_step,
            )

        # logging Utterance-level MSE, LCC, SRCC

        if mode == "dev" or mode == "vcc2018_test":
            # some evaluation-only processing, eg. decoding
            all_pred_scores = records["pred_scores"]
            all_true_scores = records["true_scores"]
            all_pred_scores = np.array(all_pred_scores)
            all_true_scores = np.array(all_true_scores)
            MSE = np.mean((all_true_scores - all_pred_scores) ** 2)
            logger.add_scalar(
                f"wav2MOS_segment_MBNet/{mode}-Utterance level MSE",
                MSE,
                global_step=global_step,
            )
            pearson_rho, _ = pearsonr(all_true_scores, all_pred_scores)
            logger.add_scalar(
                f"wav2MOS_segment_MBNet/{mode}-Utterance level LCC",
                pearson_rho,
                global_step=global_step,
            )
            spearman_rho, _ = spearmanr(all_true_scores.T, all_pred_scores.T)
            logger.add_scalar(
                f"wav2MOS_segment_MBNet/{mode}-Utterance level SRCC",
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
            all_system_true_scores = np.array(all_system_true_scores)

            MSE = np.mean((all_system_true_scores - all_system_pred_scores) ** 2)
            pearson_rho, _ = pearsonr(all_system_true_scores, all_system_pred_scores)
            spearman_rho, _ = spearmanr(all_system_true_scores, all_system_pred_scores)

            tqdm.write(f"[{mode}] System-level MSE  = {MSE:.4f}")
            tqdm.write(f"[{mode}] System-level LCC  = {pearson_rho:.4f}")
            tqdm.write(f"[{mode}] System-level SRCC = {spearman_rho:.4f}")

            logger.add_scalar(
                f"wav2MOS_segment_MBNet/{mode}-System level MSE",
                MSE,
                global_step=global_step,
            )
            logger.add_scalar(
                f"wav2MOS_segment_MBNet/{mode}-System level LCC",
                pearson_rho,
                global_step=global_step,
            )
            logger.add_scalar(
                f"wav2MOS_segment_MBNet/{mode}-System level SRCC",
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
