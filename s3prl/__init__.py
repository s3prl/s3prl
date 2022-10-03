from pathlib import Path

# HACK: SummaryWriter must be imported at the begining or else it will lead to core dumped
# This is a known issue: https://github.com/pytorch/pytorch/issues/30651
from torch.utils.tensorboard.writer import SummaryWriter

from . import problem, hub, dataset, metric, nn, task, util, dataio

with (Path(__file__).parent.resolve() / "version.txt").open() as file:
    __version__ = file.read().strip()
