import abc

from s3prl.nn import NNModule


class Task(NNModule):
    def __init__(self) -> None:
        super().__init__()

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
    def train_reduction(self, batch_results: list, on_epoch_end: bool = None):
        raise NotImplementedError

    @abc.abstractmethod
    def valid_reduction(self, batch_results: list, on_epoch_end: bool = None):
        raise NotImplementedError

    @abc.abstractmethod
    def test_reduction(self, batch_results: list, on_epoch_end: bool = None):
        raise NotImplementedError
