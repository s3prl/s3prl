import pytest
import logging

import torch.nn as nn

from s3prl import Module, init
from s3prl.interface import IOInterface, LinearInterface

logger = logging.getLogger(__name__)


class LinearIOInterfaceCorrect(IOInterface, LinearInterface):
    @classmethod
    def interface(cls, instance) -> None:
        super().interface(instance)


class CompositeModel(Module):
    @init.method
    def __init__(self, upstream: LinearIOInterfaceCorrect):
        super().__init__()


class LinearAcceptable(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.model = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.model(x)


def test_must_classmethod():
    with pytest.raises(Exception):

        class LinearIOInterfaceWrong1(IOInterface, LinearInterface):
            def interface(self, instance) -> None:
                super().interface(instance)


def test_must_take_instance_argument():
    with pytest.raises(Exception):

        class LinearIOInterfaceWrong2(IOInterface, LinearInterface):
            @classmethod
            def interface(self) -> None:
                super().interface()

        linear = LinearAcceptable(3, 4)
        CompositeModel(linear)


def test_instance_argument_must_pass_to_parent():
    with pytest.raises(Exception):

        class LinearIOInterfaceWrong3(IOInterface, LinearInterface):
            @classmethod
            def interface(self, instance) -> None:
                super().interface()

            linear = LinearAcceptable(3, 4)
            CompositeModel(linear)


def test_no_instance_output_size():
    with pytest.raises(Exception):

        class LinearUnacceptable1(nn.Module):
            def __init__(self, input_size, output_size):
                super().__init__()
                self.input_size = input_size
                self.model = nn.Linear(input_size, output_size)

            def forward(self, x):
                return self.model(x)

        linear = LinearUnacceptable1(3, 4)
        CompositeModel(linear)


def test_no_correct_instance_io():
    with pytest.raises(Exception):

        class LinearUnacceptable2(nn.Module):
            def __init__(self, input_size, output_size):
                super().__init__()
                self.input_size = input_size
                self.output_size = output_size
                self.model = nn.Linear(input_size, output_size)

            def forward(self, x):
                return self.model(x).view(-1)

        linear = LinearUnacceptable2(3, 4)
        CompositeModel(linear)


def test_correct_argument():
    linear = LinearAcceptable(3, 4)
    CompositeModel(linear)
