import functools
import importlib
import inspect
import logging
import re
import traceback
import types
from argparse import Namespace
from dataclasses import dataclass
from typing import Any

from .functools import resolve_qualname

logger = logging.getLogger(__name__)
INIT_ENTRY_INDICATOR = "INIT_ENTRY"


@dataclass
class InitConfig:
    module: str
    qualname: str
    args: list
    kwargs: dict

    def realize(self):
        """
        Return:
            module.qualname(*args, **kwargs)
        """
        module = importlib.import_module(self.module)
        qualname = self.qualname
        if re.match(r".+\.__init__", qualname):
            qualname = qualname.rsplit(".", maxsplit=1)[0]
        attr = resolve_qualname(qualname, module)
        assert callable(attr)
        result = attr(*self.args, **self.kwargs)
        return result


def set_method_decorated(func: types.FunctionType, decorated: bool):
    func._init_method_decorated = decorated


def is_method_decorated(func: types.FunctionType):
    return hasattr(func, "_init_method_decorated") and func._init_method_decorated


def method(func: types.FunctionType):
    assert inspect.isfunction(func)

    @functools.wraps(func)
    def decorated_func(*args, **kwargs):
        # execute the original init
        result = func(*args, **kwargs)

        # save init configs
        func_from_importlib = resolve_qualname(
            func.__qualname__,
            importlib.import_module(func.__module__),
        )
        if func.__name__ == "__init__":
            assert inspect.isfunction(func_from_importlib)
            obj, *args = args
        else:
            assert result is not None
            obj = result
            if inspect.ismethod(func_from_importlib):
                args = args[1:]

        if not hasattr(obj, "_init_configs"):
            obj._init_configs = []

        obj._init_configs.insert(
            0,
            serialize(
                InitConfig(
                    func.__module__,
                    func.__qualname__,
                    args,
                    kwargs,
                )
            ),
        )

        if func.__name__ != "__init__":
            return obj

    set_method_decorated(decorated_func, True)
    return decorated_func


def serialize(item: Any):
    if isinstance(item, (tuple, list)):
        new_list = []
        for subitem in item:
            new_list.append(serialize(subitem))
        item = type(item)(new_list)

    elif isinstance(item, dict):
        new_dict = {}
        for key, value in item.items():
            new_dict[key] = serialize(value)
        item = new_dict

    elif isinstance(item, InitConfig):
        item = dict(
            module=item.module,
            qualname=item.qualname,
            args=serialize(item.args),
            kwargs=serialize(item.kwargs),
        )

    elif hasattr(item, "_init_configs"):
        init_configs = item._init_configs
        assert isinstance(init_configs, list)
        assert isinstance(init_configs[0], dict)
        return {
            INIT_ENTRY_INDICATOR: serialize(item._init_configs),
        }

    return item


def _is_serialized(serialized: dict):
    return isinstance(serialized, dict) and INIT_ENTRY_INDICATOR in serialized


def _init_configs(serialized: dict):
    return serialized[INIT_ENTRY_INDICATOR]


def deserialize(item: Any):
    if _is_serialized(item):
        result = None
        for init_config_dict in _init_configs(item):
            try:
                init_config = InitConfig(
                    init_config_dict["module"],
                    init_config_dict["qualname"],
                    deserialize(init_config_dict["args"]),
                    deserialize(init_config_dict["kwargs"]),
                )
                result = init_config.realize()
            except Exception:
                logger.warning(
                    f"Failed to initialize with init config:\n{init_config_dict}"
                )
                logger.warning(traceback.format_exc())
                continue
            else:
                break

        if result is None:
            logger.warning(
                "All the initialization methods fail. Please refer to the above traceback. "
                f"The failure module:\n{item}"
            )
            raise RuntimeError

        item = result

    elif isinstance(item, (tuple, list)):
        new_list = []
        for subitem in item:
            new_list.append(deserialize(subitem))
        item = type(item)(new_list)

    elif isinstance(item, dict):
        new_dict = {}
        for key, value in item.items():
            new_dict[key] = deserialize(value)
        item = new_dict

    return item
