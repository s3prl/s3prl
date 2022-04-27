import importlib
import types
from typing import Union

import pytest

from s3prl.base.functools import resolve_qualname


class C:
    class B:
        class A:
            pass


@pytest.mark.parametrize("item", [C.B, C.B.A])
def test_resolve_qualname(item: Union[types.ModuleType, types.FunctionType]):
    qualname = item.__qualname__
    module = importlib.import_module(item.__module__)
    resolved = resolve_qualname(qualname, module)
    assert item == resolved
