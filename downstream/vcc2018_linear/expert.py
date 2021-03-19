import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
import torch
import torch.nn as nn
from torch.distributed import is_initialized
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, DistributedSampler

from .dataset import VCC18Dataset
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

        # model_cls = eval(self.modelrc["select"])
        # model_conf = self.modelrc.get(self.modelrc["select"], {})

        self.connector = nn.Linear(upstream_dim, self.modelrc["projector_dim"])
        self.model = Model(
            input_dim=self.modelrc["projector_dim"],
            clipping=self.modelrc["clipping"],
        )
        self.objective = nn.MSELoss(reduction="none")

        self.register_buffer("best_score", torch.zeros(1))

    # Interface
    def get_dataloader(self, mode):
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
        # scores = [torch.LongTensor(score) for score in scores]
        # features_len = torch.IntTensor([len(feat) for feat in features]).to(
        #     device=features[0].device
        # )
        device = features[0].device
        length = torch.Tensor([len(feat) for feat in features]).to(device)

        features = pad_sequence(features, batch_first=True)
        features = self.connector(features)
        frame_scores, uttr_scores = self.model(features, length)

        scores = scores.to(device)
        frame_loss = self.objective(frame_scores, scores[:, None])
        uttr_loss = self.objective(uttr_scores, scores).mean()
        mask = (
            torch.arange(frame_loss.size(-1)).expand_as(frame_loss).to(device)
            < length[:, None]
        ).float()
        frame_loss = (frame_loss * mask).sum(dim=-1).div(length).mean()
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
    dataframe = pd.read_csv(Path(base_path, txt_file), index_col=False)
    return dataframe
