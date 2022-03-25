import abc
from enum import Enum
from typing import Any

from torch.utils import data

from s3prl import Object


class Mode(Enum):
    FULL = 0
    METADATA = 1


_mode = Mode.FULL


def set_mode(mode: str):
    global _mode
    if isinstance(mode, Mode):
        _mode = mode
    elif mode == "FULL":
        _mode = Mode.FULL
    elif mode == "METADATA":
        _mode = Mode.METADATA


def in_metadata_mode():
    return _mode == Mode.METADATA


class metadata_mode:
    def __init__(self):
        self.prev = None

    def __enter__(self):
        self.prev = _mode
        set_mode(Mode.METADATA)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        set_mode(self.prev)


class Dataset(Object, data.Dataset):
    def __init__(self):
        super().__init__()

    @staticmethod
    def in_metadata_mode():
        return in_metadata_mode()

    @abc.abstractmethod
    def collate_fn(self, samples):
        pass
