from pathlib import Path as _Path

from . import corpus, dataset, encoder, metric, nn, problem, sampler, task, util

with (_Path(__file__).parent.resolve() / "version.txt").open() as file:
    __version__ = file.read().strip()
