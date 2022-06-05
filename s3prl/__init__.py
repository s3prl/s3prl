from pathlib import Path as _Path

from .base import (
    Container,
    LogData,
    LogDataType,
    Logs,
    Module,
    Object,
    Output,
    cache,
    field,
    init,
    set_use_cache,
)
from .util.workspace import Workspace

with (_Path(__file__).parent.resolve() / "version.txt").open() as file:
    __version__ = file.read().strip()
