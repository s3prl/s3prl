"""Downstream expert for Query-by-Example Spoken Term Detection on QUESST 2014."""

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from lxml import etree
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from .dataset import SWS2013Dataset
from .model import Model


class DownstreamExpert(nn.Module):
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log
    """

    def __init__(
        self, upstream_dim: int, downstream_expert: dict, expdir: str, **kwargs
    ):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.datarc = downstream_expert["datarc"]
        self.modelrc = downstream_expert["modelrc"]
        self.expdir = Path(expdir)
        self.train_dataset = SWS2013Dataset(**self.datarc)

        self.model = Model(
            input_dim=upstream_dim,
            **self.modelrc,
        )
        self.objective = nn.CosineEmbeddingLoss()

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
        return DataLoader(
            self.train_dataset,
            sampler=WeightedRandomSampler(
                self.train_dataset.sample_weights,
                len(self.train_dataset.sample_weights),
            ),
            batch_size=self.datarc["batch_size"],
            drop_last=True,
            num_workers=self.datarc["num_workers"],
            collate_fn=self.train_dataset.collate_fn,
        )

    # Interface
    def get_dev_dataloader(self):
        return None

    # Interface
    def get_test_dataloader(self):
        return None

    # Interface
    def forward(
        self,
        features,
        labels,
        records,
        **kwargs,
    ):
        audio_tensors = torch.stack(features[: len(features) // 2])
        query_tensors = torch.stack(features[len(features) // 2 :])
        labels = torch.stack(labels).to(audio_tensors.device)
        audio_embs = self.model(audio_tensors)
        query_embs = self.model(query_tensors)
        return self.objective(audio_embs, query_embs, labels)

    # interface
    def log_records(self, records, **kwargs):
        """Perform DTW and save results."""
        pass
