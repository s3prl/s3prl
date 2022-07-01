import torch
import torch.nn as nn

from s3prl.base import Logs
from s3prl.base.output import Output
from s3prl.base.workspace import Workspace
from s3prl.util import workspace
from s3prl.util.workspace import Workspace

from .base import Task


class DumpFeature(Task):
    def __init__(self, model: nn.Module, **kwds) -> None:
        super().__init__()
        self.model = model

    @torch.no_grad()
    def forward(self, split: str, x, x_len, unique_name, workspace: Workspace, **kwds):
        self.model.eval()
        feats, feats_len = self.model(x, x_len).slice(2)
        feats = torch.stack(
            feats, dim=1
        )  # (batch_size, num_layer, seqlen, hidden_size)

        feat_dir = workspace / "feat"
        for feat, feat_len, name in zip(feats, feats_len, unique_name):
            feat = feat[:, :feat_len, :]
            feat = feat.detach().cpu().numpy()
            feat_dir.put(feat, name, "npy")
        return Output()

    def reduction(
        self, split: str, batch_results: list, on_epoch_end: bool = None, **kwds
    ):
        return Output(
            logs=Logs(),
        )
