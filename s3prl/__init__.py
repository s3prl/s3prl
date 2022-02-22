from .base import init, Object, Output
from .nn import Module
from .task import Task

with open("version.txt") as file:
    __version__ = file.read().strip()
