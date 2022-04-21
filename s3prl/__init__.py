from pathlib import Path as _Path

from .base import (
    init,
    cache,
    set_use_cache,
    Object,
    Container,
    Output,
    Module,
    Logs,
    LogData,
    LogDataType,
)

with (_Path(__file__).parent.parent / "version.txt").open() as file:
    __version__ = file.read().strip()
