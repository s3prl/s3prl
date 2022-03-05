from __future__ import annotations
import abc
import functools
from s3prl import Module, Output


class NNModule(Module):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(self) -> Output:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def input_size(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def output_size(self):
        raise NotImplementedError
