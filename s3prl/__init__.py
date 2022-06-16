from pathlib import Path as _Path

from . import nn
from .base import *

with (_Path(__file__).parent.resolve() / "version.txt").open() as file:
    __version__ = file.read().strip()
