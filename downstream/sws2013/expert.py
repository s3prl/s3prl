"""Downstream expert for Query-by-Example Spoken Term Detection on SWS 2013."""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

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

    # Interface
    def get_dataloader(self, mode):
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
    def forward(
        self,
        mode,
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
        loss = self.objective(audio_embs, query_embs, labels)

        records["loss"].append(loss.item())

        return loss

    # interface
    def log_records(self, mode, records, logger, global_step, **kwargs):
        prefix = f"sws2013/{mode}"
        for key, val in records.items():
            average = sum(val) / len(val)
            logger.add_scalar(f'{prefix}-{key}', average, global_step=global_step)
