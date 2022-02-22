import torch
import torch.nn as nn
from s3prl.base.interface import Interface


class NNInterface(Interface):
    def __init__(self):
        super().__init__()

    @classmethod
    def interface(cls, instance) -> None:
        super().interface(instance)
        assert isinstance(instance, nn.Module)


class IOInterface(NNInterface):
    def __init__(self):
        super().__init__()
        self.input_size = None
        self.output_size = None

    @classmethod
    def interface(cls, instance) -> None:
        super().interface(instance)
        assert isinstance(instance.input_size, int)
        assert isinstance(instance.output_size, int)


class LinearInterface(NNInterface):
    def __init__(self):
        super().__init__()

    @classmethod
    def interface(cls, instance) -> None:
        super().interface(instance)
        rand = torch.randn(32, 100, instance.input_size)
        x = instance(rand)
        assert x.shape == (32, 100, instance.output_size)
