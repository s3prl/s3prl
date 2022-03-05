from .base import init, cache, Object, Output, Module, Logs
from .nn import NNModule
from .task import Task

with open("version.txt") as file:
    __version__ = file.read().strip()
