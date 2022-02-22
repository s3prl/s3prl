import tempfile

from s3prl import init
from s3prl.nn import Module
from s3prl.nn import FrameLevelLinear as Linear


class AnyChild(Module):
    @init.method
    def __init__(
        self, model: Module, *modules, save=False, clear=None, **kwargs
    ) -> None:
        super().__init__()
        self.model = model

    @property
    def input_size(self):
        return self.model.input_size

    @property
    def output_size(self):
        return self.model.output_size

    def forward(self, x):
        return self.model(x)


def test_Linear(helpers):
    linear = Linear(3, 4)
    helpers.validate_module(linear)


def test_AnyChild(helpers):
    linear = Linear(3, 4)
    any = AnyChild(linear)
    helpers.validate_module(any)

def test_state_dict():
    linear = Linear(3, 4)
    any = AnyChild(linear)
    any.exclude_from_state_dict(["model"])
    states = any.state_dict()
    assert len(states) == 0
    any.load_state_dict(states)

def test_checkpoint(helpers):
    linear = Linear(3, 4)
    any = AnyChild(linear)

    any.exclude_from_state_dict(["model"])
    checkpoint = any.checkpoint()
    new_any = Module.from_checkpoint(checkpoint)
    helpers.is_same_module(any, new_any)

    any.include_to_state_dict(["model"])
    checkpoint = any.checkpoint()
    new_any = Module.from_checkpoint(checkpoint)
    helpers.is_same_module(any, new_any)

def test_save_checkpoint(helpers):
    linear = Linear(3, 4)
    any = AnyChild(linear)

    with tempfile.NamedTemporaryFile() as f:
        any.exclude_from_state_dict(["model"])
        any.save_checkpoint(f.name)
        new_any = Module.load_checkpoint(f.name)
        helpers.is_same_module(any, new_any)

    with tempfile.NamedTemporaryFile() as f:
        any.include_to_state_dict(["model"])
        any.save_checkpoint(f.name)
        new_any = Module.load_checkpoint(f.name)
        helpers.is_same_module(any, new_any)
