import logging
from typing import Any

import torch
import pytest
import torch.optim as optim

from s3prl import init
from s3prl.nn import Module

logger = logging.getLogger(__name__)


class Helper:
    @classmethod
    def validate_object(cls, obj: Any):
        serialized = init.serialize(obj)
        new_obj = init.deserialize(serialized)
        new_serialized = init.serialize(new_obj)
        assert serialized == new_serialized
        return new_obj

    @classmethod
    def pseudo_output(cls, obj: object):
        if isinstance(obj, torch.Tensor):
            return obj
        if isinstance(obj, (tuple, list)):
            return cls.pseudo_output(obj[0])
        elif isinstance(obj, dict):
            return cls.pseudo_output(list(obj.values())[0])
        else:
            logger.error(f"Unexpected obj type: {obj}")
            raise NotImplementedError

    @classmethod
    def validate_module(cls, module: Module):
        optimizer = optim.Adam(module.parameters(), lr=1e-3)
        x = torch.randn(32, module.input_size)
        y = cls.pseudo_output(module(x))

        loss = y.sum()
        loss.backward()
        optimizer.step()

        new_module = cls.validate_object(module)
        new_module.load_state_dict(module.state_dict())
        assert cls.is_same_module(module, new_module)
        return new_module

    @classmethod
    def is_same_module(cls, module1: Module, module2: Module):
        rand_input = torch.randn(module1.input_size)
        output1 = cls.pseudo_output(module1(rand_input))
        output2 = cls.pseudo_output(module2(rand_input))
        return torch.allclose(output1, output2)


@pytest.fixture
def helpers():
    return Helper
