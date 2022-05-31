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
    init,
    set_use_cache,
)

with (_Path(__file__).parent.resolve() / "version.txt").open() as file:
    __version__ = file.read().strip()
