from pathlib import Path as _Path

from . import dataset, metric, nn, problem, task, util

with (_Path(__file__).parent.resolve() / "version.txt").open() as file:
    __version__ = file.read().strip()
