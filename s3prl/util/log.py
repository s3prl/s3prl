from enum import Enum
from typing import Any
from dataclasses import dataclass


class LogDataType(Enum):
    SCALAR = 0
    TEXT = 1
    AUDIO = 2
    IMAGE = 3
    TENSOR = 4


@dataclass
class Log:
    name: str
    data: Any
    data_type: LogDataType
