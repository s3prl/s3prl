from pathlib import Path as _Path

from s3prl.base import *

from . import (
    nn,
    problem,
    corpus,
    dataset,
    sampler,
    encoder,
    metric,
    task,
    util,
)

with (_Path(__file__).parent.resolve() / "version.txt").open() as file:
    __version__ = file.read().strip()
