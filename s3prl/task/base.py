import abc

from s3prl import Module, init


class Task(Module):
    @init.method
    def __init__(self) -> None:
        super().__init__()

    @abc.abstractmethod
    def forward(self):
        raise NotImplementedError

    @abc.abstractmethod
    def inference(self):
        raise NotImplementedError

    @abc.abstractmethod
    def training_step(self):
        raise NotImplementedError

    @abc.abstractmethod
    def validation_step(self):
        raise NotImplementedError

    @abc.abstractmethod
    def test_step(self):
        raise NotImplementedError

    @abc.abstractmethod
    def training_reduction(self, batch_results: list, on_epoch_end: bool = None):
        return self._general_reduction(batch_results, on_epoch_end)

    @abc.abstractmethod
    def validation_reduction(self, batch_results: list, on_epoch_end: bool = None):
        return self._general_reduction(batch_results, on_epoch_end)

    @abc.abstractmethod
    def test_reduction(self, batch_results: list, on_epoch_end: bool = None):
        return self._general_reduction(batch_results, on_epoch_end)
