"""
Dump feature Task

Authors
  * Yist Y. Lin 2021
  * Leo 2022
"""

from pathlib import Path

import torch
import torch.nn as nn

from .base import Task

__all__ = ["DumpFeature"]


class DumpFeature(Task):
    def __init__(self, model: nn.Module, dump_feat_dir: str = "feat") -> None:
        super().__init__()
        self.model = model
        self.dump_feat_dir = dump_feat_dir

    @torch.no_grad()
    def forward(self, split: str, x, x_len, unique_name, _dump_dir: str):
        self.model.eval()
        feats, feats_len = self.model(x, x_len).slice(2)
        feats = torch.stack(
            feats, dim=1
        )  # (batch_size, num_layer, seqlen, hidden_size)

        feat_dir = Path(_dump_dir) / self.dump_feat_dir
        for feat, feat_len, name in zip(feats, feats_len, unique_name):
            feat = feat[:, :feat_len, :]
            feat = feat.detach().cpu()
            torch.save(feat, str(feat_dir / f"{name}.pt"))

        pseudo_loss = torch.zeros(1, requires_grad=True)
        return pseudo_loss, {}

    def reduction(self, split: str, batch_results: list, _dump_dir: str):
        return {}
