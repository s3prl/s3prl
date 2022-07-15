from __future__ import annotations

import importlib
import logging
from collections import OrderedDict
from copy import deepcopy
from functools import partial
from types import FunctionType
from typing import Any, Callable, List, Union

import torch
import yaml

logger = logging.getLogger(__name__)


def _qualname_to_cls(qualname: str):
    # TODO
    # This assume that the qualname does not have dot
    # Hence it can't support classmethod, whose qualname is {class_name}.{method_name}
    module_name, cls_name = qualname.rsplit(".", maxsplit=1)
    cls = getattr(importlib.import_module(module_name), cls_name)
    return cls


class field:
    def __init__(
        self,
        value,
        description: str,
        dtype: type = None,
        check_fn: Callable[[Any], bool] = None,
    ) -> None:
        self.value = value
        self.dtype = dtype
        self.description = description
        self.check_fn = check_fn or (lambda x: True)

    @classmethod
    def sanitize(cls, obj):
        """
        Ensure the output of this function is not wrapped by **field**
        """
        if isinstance(obj, __class__):
            return obj.value
        else:
            return obj

    def update(self, value):
        if not self.check_fn(value):
            raise ValueError(
                f"The input value {value} does not pass the check function {self.check_fn}: {self.description}"
            )
        self.value = value

    def __str__(self):
        return self.show(None)

    def __repr__(self):
        return f"{__class__.__name__}(value={self.value}, dtype={self.dtype}), description={self.description}, check_fn={self.check_fn}"

    def show(self, show_value_func: callable = None):
        show_value_func = show_value_func or str
        basic = show_value_func(self.value)

        basic += "    [COMMENT] "
        if self.dtype is not None:
            if not isinstance(self.dtype, type):
                basic += f"({self.dtype}) "
            elif self.dtype.__module__ == "builtins":
                basic += f"({self.dtype.__qualname__}) "
            else:
                basic += f"({self.dtype.__module__}.{self.dtype.__qualname__}) "

        message = (
            self.description.replace("\n", " [NEWLINE] ")
            .replace(":", "[DOT]")
            .replace(" ", "[SPACE]")
        )
        basic += message
        return basic


class Container(OrderedDict):
    """
    You can use Container just like a normal dictionary (with some
    augmented functions):

    .. code-block:: python

        obj = Container({"a": 3, "b": 4})

        # or

        obj = Container(a=3, b=4)

    The core function of the Container class is to provide the
    easy-to-use and intuitive interface for manipulating the returning
    dictionary/namespace which can have more key-value pairs in the
    future. In the following examples, we use the 'result' for
    demonstration.

    .. code-block:: python

        result = Container(output=3, loss=4)

    The core usages of Container are:

    1. Get value as a Namespace.

    .. code-block:: python

        assert result.output == result["output"] == 3

    2. Get value by index. The OrderedDict is ordered-sensitive with
    the initialization arguments. That is:

    .. code-block:: python

        Container(output=3, loss=4) != Container(loss=4, output=3)

    On the other hand, in common machine learning we don't use integer
    as keys which are not informative. Hence, we can use integers as the
    ordered keys to access the ordered values:

    .. code-block:: python

        assert 3 == result[0] == result.output
        assert 4 == result[1] == result.loss

    3. To get a subset of the Container as a smaller version of
    Container or as a tuple containing only values to act like
    the conventional function return

    .. code-block:: python

        assert 3 == result.subset("output")
        assert (4, 3) == result.subset("loss", "output")
        assert (4, 3) == result.subset(1, 0)
        assert 4 == result.subset(1)
        assert Container(loss=4) == result.subset("loss", as_type="dict")

    4. Similar to 3, but leverage the 2. function to enable slicing.

    .. code-block:: python

        assert 3 == result.slice(1)
        assert (3, 4) == result.slice(2)
        assert 4 == result.slice(1, 2)
        assert Container(loss=4) == result.slice(1, 2, as_type="dict")

    Important note for the functions 2. and 4.
    The length of a Container object can change across S3PRL versions.
    Please must NOT slice or subset the Container object with negative
    index or the index derived from dynamically obtained __len__ function.
    This will lead to the breaking change if the later version
    """

    _reserved_keys = [
        "_normalize_key",
        "update",
        "override",
        "add",
        "detach",
        "cpu",
        "to",
        "subset",
        "slice",
        "split",
        "select",
        "deselect",
        "to_dict",
        "clone",
    ]

    UNFILLED_PATTERN = "???"
    QUALNAME_PATTERNS = ["_cls", "CLS"]

    def check_no_unfilled_field(self):
        unfilled_fields = self.unfilled_fields()
        if len(unfilled_fields) > 0:
            raise ValueError(
                "There are unfilled but required fields in "
                f"the config:\n\n{unfilled_fields}"
            )

    def unfilled_fields(self):
        from s3prl.util.override import parse_overrides

        unfilleds = self.list_unfilled_fields()
        override = []
        for field in unfilleds:
            override += [f"--{field}", f"{__class__.UNFILLED_PATTERN}"]
        override_dict = __class__(parse_overrides(override))
        override_dict.update(self, True, False)
        return override_dict

    def list_unfilled_fields(self):
        unfilleds = []
        self._unfilled_fields(self, "", unfilleds)
        return unfilleds

    @classmethod
    def _unfilled_fields(cls, obj, parent: str, unfilleds: list):
        if isinstance(obj, (tuple, list)):
            for idx, item in enumerate(obj):
                cls._unfilled_fields(item, f"{parent}[{idx}].", unfilleds)
        elif isinstance(obj, dict):
            for key, item in obj.items():
                cls._unfilled_fields(item, f"{parent}{key}.", unfilleds)
        elif isinstance(obj, str) and __class__.UNFILLED_PATTERN in obj:
            unfilleds.append(parent[:-1])
        elif isinstance(obj, field):
            if isinstance(obj.value, str) and __class__.UNFILLED_PATTERN in obj.value:
                unfilleds.append(parent[:-1])

    def kwds(self):
        new_copy = self.clone()
        self._no_cls(new_copy)
        return new_copy

    def _instantiate(self, *args, **kwds):
        new_self = deepcopy(self)
        new_self = new_self.extract_fields()

        cls = None
        for pattern in __class__.QUALNAME_PATTERNS:
            if pattern in new_self:
                assert cls is None, f"Duplicated keys for cls: {list(new_self.keys())}"
                cls = new_self[pattern]

        assert callable(cls)
        effective_kwds = new_self.kwds().override(kwds)
        return cls(*args, **effective_kwds)

    def __call__(self, *args, **kwds: dict) -> Any:
        """
        Calling self._cls with other fields as **kwds
        The positional arguments should be provided by *args
        """
        return self._instantiate(*args, **kwds)

    @classmethod
    def _no_cls(cls, obj):
        if isinstance(obj, dict):
            for key in list(obj.keys()):
                if key in cls.QUALNAME_PATTERNS:
                    obj.pop(key)

    def cls_fields(self, holder: list = None):
        holder = []
        self._cls_fields(self, "", holder)
        return holder

    @classmethod
    def _cls_fields(cls, obj, parent: str, clses: list):
        if isinstance(obj, (tuple, list)):
            for idx, item in enumerate(obj):
                cls._cls_fields(item, f"{parent}[{idx}].", clses)
        elif isinstance(obj, dict):
            for key, item in obj.items():
                if key in __class__.QUALNAME_PATTERNS:
                    clses.append((parent[:-1], item))
                else:
                    cls._cls_fields(item, f"{parent}{key}.", clses)

    def indented_str(self, indent: str):
        ori_str = str(self)
        indented_lines = [f"{indent}{line}" for line in ori_str.split("\n")]
        return "\n".join(indented_lines)

    def clone(self) -> Container:
        return deepcopy(self)

    def _normalize_key(self, k):
        if isinstance(k, int):
            return list(super().keys())[k]
        return k

    def override(self, new_dict: dict):
        """
        Nestedly update old dict with new dict. Allow duplicate.
        """
        self.update(new_dict, override=True)
        return self

    def add(self, new_dict: dict):
        """
        Nestedly update old dict with new dict. Not allow duplicate.
        """
        self.update(new_dict, override=False)
        return self

    def update(self, new_dict: dict, override: bool, create: bool = True):
        for key, value in new_dict.items():
            if not create and key not in self:
                continue

            if isinstance(value, dict):
                if key not in self:
                    self.__setitem__(key, __class__())
                target = self.__getitem__(key)
                if isinstance(target, dict) and not isinstance(target, Container):
                    target = Container(target)
                    self.__setitem__(key, target)
                if isinstance(target, __class__) and not isinstance(value, newdict):
                    target.update(value, override, create)
                else:
                    self.__setitem__(key, value)
            else:
                if hasattr(self, key):
                    if not override:
                        old_value = self.__getitem__(key)
                        if value != old_value and id(value) != id(old_value):
                            logger.warning(
                                f"Old v.s. new dict have a duplicated key {key} with values {old_value} v.s. {value}"
                            )
                            raise ValueError(
                                f"override option is false. {key} exists in the original dict"
                            )
                self.__setitem__(key, value)
            if key.startswith("_"):
                # special keys like "_cls" and "_name" should be at the first
                self.move_to_end(key, last=False)

    def __getitem__(self, k):
        k = self._normalize_key(k)
        return super().__getitem__(k)

    @staticmethod
    def deserialize_cls(key):
        from s3prl.util import registry

        if key == __class__.UNFILLED_PATTERN:
            return key

        cls = key
        if not (isinstance(key, type) or callable(key)):
            try:
                cls = _qualname_to_cls(key)
            except:
                try:
                    cls = registry.get(key)
                except:
                    logger.error(f"Cannot resolve _cls = {key}. Might be an error.")
        return cls

    @staticmethod
    def serialize_cls(cls):
        from s3prl.util import registry

        key = cls
        try:
            if registry.contains(cls):
                key = registry.serialize(cls)
            else:
                key = f"{cls.__module__}.{cls.__qualname__}"
        except:
            pass
        return key

    def __setitem__(self, k, v, replace_field=False) -> None:
        k = self._normalize_key(k)
        assert k not in self._reserved_keys, f"'{k}' cannot be used"
        if type(v) == dict:
            v = __class__(v)
        if k in __class__.QUALNAME_PATTERNS:
            if isinstance(v, field):
                cls = self.deserialize_cls(v.value)
                v.value = cls
            else:
                v = self.deserialize_cls(v)
        if (
            k in self
            and isinstance(self[k], field)
            and not isinstance(v, field)
            and not replace_field
        ):
            self[k].update(v)
        else:
            super().__setitem__(k, v)

    def to_dict(self, must_invertible=True):
        return self.to_dict_impl(self, must_invertible)

    @classmethod
    def to_dict_impl(cls, dictionary: Container, must_invertible=True):
        result = dict()
        for k, v in dictionary.items():
            if isinstance(v, dict):
                v = cls.to_dict_impl(v, must_invertible)
            elif k in __class__.QUALNAME_PATTERNS:
                if isinstance(v, field):
                    if must_invertible:
                        v = v.value
                    else:
                        v = v.show(cls.serialize_cls)
                else:
                    v = cls.serialize_cls(v)
            elif isinstance(v, field):
                if must_invertible:
                    v = v.value
                else:
                    v = v.show()
            result[k] = v
        return result

    def __str__(self) -> str:
        return yaml.dump(
            self.to_dict(must_invertible=False), sort_keys=False, width=float("inf")
        )

    def __repr__(self) -> str:
        return f"Container({str(self.to_dict())})"

    def __getattribute__(self, name: str) -> Any:
        keys = super().keys()
        if name in keys:
            return self.__getitem__(name)
        return super().__getattribute__(name)

    def __delattr__(self, name: str) -> None:
        keys = super().keys()
        if name in keys:
            return self.__delitem__(name)
        return super().__delattr__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        return self.__setitem__(name, value)

    def detach(self):
        self._recursive_apply(self, self._detach_impl)
        return self

    @staticmethod
    def _detach_impl(obj):
        return obj.detach() if isinstance(obj, torch.Tensor) else obj

    def cpu(self):
        self._recursive_apply(self, self._cpu_impl)
        return self

    @staticmethod
    def _cpu_impl(obj):
        return obj.cpu() if isinstance(obj, torch.Tensor) else obj

    def to(self, target):
        self._recursive_apply(self, partial(self._to_impl, target=target))
        return self

    @staticmethod
    def _to_impl(obj, target):
        return obj.to(target) if isinstance(obj, torch.Tensor) else obj

    def extract_fields(self):
        self._extract_field_impl(self)
        return self

    @classmethod
    def _extract_field_impl(cls, obj):
        if isinstance(obj, field):
            return obj.value
        elif isinstance(obj, list):
            for index in range(len(obj)):
                obj[index] = cls._extract_field_impl(obj[index])
        elif isinstance(obj, __class__):
            for key in list(obj.keys()):
                if isinstance(obj[key], field):
                    replace_field = True
                    value = obj[key].value
                else:
                    replace_field = False
                    value = obj[key]
                obj.__setitem__(key, cls._extract_field_impl(value), replace_field)
        elif isinstance(obj, dict):
            for key in list(obj.keys()):
                obj[key] = cls._extract_field_impl(obj[key])
        return obj

    @classmethod
    def _recursive_apply(cls, obj, apply_fn):
        if isinstance(obj, list):
            for index in range(len(obj)):
                obj[index] = cls._recursive_apply(obj[index], apply_fn)
        elif isinstance(obj, dict):
            for key in list(obj.keys()):
                obj[key] = cls._recursive_apply(obj[key], apply_fn)
        else:
            return apply_fn(obj)
        return obj

    def subset(
        self,
        *names: List[Union[str, int]],
        as_type: str = "tuple",
        exclude: bool = False,
    ):
        """
        Args:
            as_type (str): "tuple" or "dict"
            exclude (bool): the "deselect" the names and leave the remaining
        """

        keys = list(super().keys())
        names = [keys[name] if isinstance(name, int) else name for name in names]
        if exclude:
            names = [key for key in keys if key not in names]
        result = __class__(**{name: self.__getitem__(name) for name in names})
        if as_type == "dict":
            return result
        result = tuple(result.values())
        return result if len(result) > 1 else result[0]

    def slice(
        self,
        start_or_end: int,
        end: int = None,
        step: int = 1,
        as_type: str = "tuple",
    ):
        """
        Args:
            as_type (str): "tuple" or "dict"

        The usage is similar to the range() built-in function.
        However, please must NOT use negative index (counted from
        the end) or the index derived by __len__ dynamically.

        Correct Examples:
            self.slice(2): Get front 2 values
            self.slice(1, 3): Get values of index 1, 2
            self.slice(1, 5, 2): Get values of 1, 3

        Wrong Examples: these usages can break after upgrading to newer S3PRL
            self.slice(2, -1),
            self.slice(2, len(self)),
            self.slice(2, len(self.keys())),
        """
        if end is None:
            start = 0
            end = start_or_end
        else:
            start = start_or_end
            end = end

        assert isinstance(start, int) and start >= 0
        assert isinstance(end, int) and end >= 0

        interval = list(range(start, end, step))
        return self.subset(*interval, as_type=as_type)

    def split(self, start_or_end: int, end: int = None, step: int = 1):
        selected = self.slice(start_or_end, end, step, as_type="dict")
        deselected = self.deselect(*list(selected.keys()))
        return [*list(selected.values()), deselected]

    def select(self, *names: List[str]):
        return self.subset(*names, as_type="dict")

    def deselect(self, *names: List[str]):
        return self.subset(*names, as_type="dict", exclude=True)


class newdict(Container):
    pass
