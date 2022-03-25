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

with open("version.txt") as file:
    __version__ = file.read().strip()
