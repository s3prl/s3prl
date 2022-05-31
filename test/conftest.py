import logging
import tempfile
from typing import Any

import pytest
import torch
import torch.optim as optim

from s3prl import Module, Object, init

logger = logging.getLogger(__name__)


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--runextra", action="store_true", help="run tests with extra dependencies"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line(
        "markers", "extra_dependency: mask test requiring extra dependencies to run"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)

    if not config.getoption("--runextra"):
        skip_extra = pytest.mark.skip(reason="need --runextra option to run")
        for item in items:
            if "extra_dependency" in item.keywords:
                item.add_marker(skip_extra)


class Helper:
    @classmethod
    def validate_object(cls, obj: Any):
        serialized = init.serialize(obj)
        new_obj = init.deserialize(serialized)
        new_serialized = init.serialize(new_obj)
        assert serialized == new_serialized
        return new_obj

    @classmethod
    def get_single_tensor(cls, obj: Object):
        if isinstance(obj, torch.Tensor):
            return obj
        if isinstance(obj, (tuple, list)):
            sanitized = []
            for o in obj:
                sanitized.append(cls.get_single_tensor(o))
            sanitized = [s for s in sanitized if s is not None]
            return sanitized[0] if len(sanitized) > 0 else None
        elif isinstance(obj, dict):
            return cls.get_single_tensor(list(obj.values()))
        else:
            return None

    @classmethod
    def validate_module(cls, module: Module, *args, device="cpu", **kwargs):
        optimizer = optim.Adam(module.parameters(), lr=1e-3)
        y = cls.get_single_tensor(module(*args, **kwargs))

        loss = y.sum()
        loss.backward()
        optimizer.step()

        with tempfile.NamedTemporaryFile() as file:
            module.save_checkpoint(file.name)
            module_reload = Object.load_checkpoint(file.name).to(device)

        assert cls.is_same_module(module, module_reload, *args, **kwargs)
        return module_reload

    @classmethod
    def is_same_module(cls, module1: Module, module2: Module, *args, **kwargs):
        module1.eval()
        module2.eval()
        with torch.no_grad():
            output1 = module1(*args, **kwargs)
            output2 = module2(*args, **kwargs)
        tensor1 = cls.get_single_tensor(output1)
        tensor2 = cls.get_single_tensor(output2)
        return torch.allclose(tensor1, tensor2)


@pytest.fixture
def helpers():
    return Helper
