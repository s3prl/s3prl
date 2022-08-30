import abc
from collections import defaultdict
from typing import List

import torch


class Task(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def get_state(self):
        return {}

    def set_state(self, state: dict):
        pass

    def dict_of_lists(self, list_of_dicts: List[dict]):
        keys = list(list_of_dicts[0].keys())
        dol = defaultdict(list)
        for d in list_of_dicts:
            assert sorted(keys) == sorted(list(d.keys()))
            for k, v in d.items():
                if isinstance(v, (tuple, list)):
                    dol[k].extend(v)
                else:
                    dol[k].append(v)
        return dict(dol)

    @abc.abstractmethod
    def predict(self):
        raise NotImplementedError

    def forward(self, mode: str, *args, **kwargs):
        return getattr(self, f"{mode}_step")(*args, **kwargs)

    def reduction(self, mode: str, *args, **kwargs):
        return getattr(self, f"{mode}_reduction")(*args, **kwargs)

    @abc.abstractmethod
    def train_step(self):
        raise NotImplementedError

    @abc.abstractmethod
    def valid_step(self):
        raise NotImplementedError

    @abc.abstractmethod
    def test_step(self):
        raise NotImplementedError

    @abc.abstractmethod
    def train_reduction(self, batch_results: list):
        raise NotImplementedError

    @abc.abstractmethod
    def valid_reduction(self, batch_results: list):
        raise NotImplementedError

    @abc.abstractmethod
    def test_reduction(self, batch_results: list):
        raise NotImplementedError
