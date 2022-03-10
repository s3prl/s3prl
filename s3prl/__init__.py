from .base import init, cache, Object, Output, Module, Logs, LogData, LogDataType
from .nn import NNModule
from .task import Task
from .dataset import Dataset

with open("version.txt") as file:
    __version__ = file.read().strip()
