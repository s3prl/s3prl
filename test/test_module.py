import tempfile

import torch
import torch.nn as nn

from s3prl import Module
from s3prl.nn import FrameLevel as Linear


class AnyChild(Module):
    def __init__(
        self, model: Module, *modules, save=False, clear=None, **kwargs
    ) -> None:
        super().__init__()
        self.model = model
        self.linear = nn.Linear(3, 4)

    @property
    def input_size(self):
        return self.model.input_size

    @property
    def output_size(self):
        return self.model.output_size

    def forward(self, x, x_len=None):
        return self.model(x, x_len)


def test_Linear(helpers):
    linear = Linear(3, 4)
    helpers.validate_module(linear, torch.randn(32, linear.input_size))


def test_AnyChild(helpers):
    linear = Linear(3, 4)
    any = AnyChild(linear)
    helpers.validate_module(any, torch.randn(32, any.input_size))


def test_state_dict():
    linear = Linear(3, 4)
    any = AnyChild(linear)
    states = any.state_dict()
    assert len(states) == 6
    any.load_state_dict(states)


def test_checkpoint(helpers):
    linear = Linear(3, 4)
    any = AnyChild(linear)

    checkpoint = any.checkpoint()
    new_any = Module.from_checkpoint(checkpoint)
    helpers.is_same_module(any, new_any, torch.randn(32, any.input_size))

    checkpoint = any.checkpoint()
    new_any = Module.from_checkpoint(checkpoint)
    helpers.is_same_module(any, new_any, torch.randn(32, any.input_size))


def test_save_checkpoint(helpers):
    linear = Linear(3, 4)
    any = AnyChild(linear)

    with tempfile.NamedTemporaryFile() as f:
        any.exclude_from_state_dict("model")
        any.save_checkpoint(f.name)
        new_any = Module.load_checkpoint(f.name)
        helpers.is_same_module(any, new_any, torch.randn(32, any.input_size))

    with tempfile.NamedTemporaryFile() as f:
        any.include_to_state_dict("model")
        any.save_checkpoint(f.name)
        new_any = Module.load_checkpoint(f.name)
        helpers.is_same_module(any, new_any, torch.randn(32, any.input_size))
