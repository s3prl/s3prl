from s3prl.base.output import Output
from s3prl.util import workspace
from .base import Task
import torch
import torch.nn as nn
from s3prl.util.workspace import Workspace
from s3prl.base import Logs


class DumpFeature(Task):
    def __init__(self, model: nn.Module, workspace: Workspace, **kwds) -> None:
        super().__init__()
        self.model = model
        self.feat_dir = workspace / "feat"

    @torch.no_grad()
    def forward(self, split: str, x, x_len, unique_name, **kwds):
        self.model.eval()
        feats, feats_len = self.model(x, x_len).slice(2)
        for feat, feat_len, name in zip(feats, feats_len, unique_name):
            feat = feat[:feat_len]
            feat = feat.detach().cpu().numpy()
            self.feat_dir.put(feat, name, "npy")
        return Output()

    def reduction(self, split: str, batch_results: list, on_epoch_end: bool = None):
        return Output(
            logs=Logs(),
        )
