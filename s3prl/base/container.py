from functools import partial
from typing import Any, List, Union
from collections import OrderedDict

import torch


class Container(OrderedDict):
    """
    The core function of the Container class is to provide the
    easy-to-use and intuitive interface for manipulating the returning
    dictionary/namespace which can have more key-value pairs in the
    future. In the following examples, we use the 'result' for
    demonstration.

    0.
    Assume:
    result = Container(output=3, loss=4)

    The core usages of Container are:

    1.
    Get value as a Namespace. Personally, I don't like to see the
    ["something"] syntax which is very noisy. Hence, the following
    is possible

    assert result.output == result["output"] == 3

    2.
    Get value by index. The OrderedDict is ordered-sensitive with
    the initialization arguments. That is:

    Container(output=3, loss=4) != Container(loss=4, output=3)

    On the other hand, in common machine learning we don't use integer
    as keys which are not informative. Hence, we can use integers as the
    ordered keys to access the ordered values:

    assert 3 == result[0] == result.output
    assert 4 == result[1] == result.loss

    3.
    To get a subset of the Container as a smaller version of
    Container or as a tuple containing only values to act like
    the conventional function return

    assert 3 == result.subset("output")
    assert (4, 3) == result.subset("loss", "output")
    assert (4, 3) == result.subset(1, 0)
    assert 4 == result.subset(1)
    assert Container(loss=4) == result.subset("loss", as_type="dict")

    4.
    Similar to 3, but leverage the 2. function to enable slicing.

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
    ]

    def _normalize_key(self, k):
        if isinstance(k, int):
            return list(super().keys())[k]
        return k

    def override(self, new_dict: dict):
        """
        Nestedly update old dict with new dict. Allow duplicate.
        """
        self.update(new_dict, override=True)

    def add(self, new_dict: dict):
        """
        Nestedly update old dict with new dict. Not allow duplicate.
        """
        self.update(new_dict, override=False)

    def update(self, new_dict: dict, override: bool):
        for key, value in new_dict.items():
            if isinstance(value, dict):
                if key not in self:
                    self.__setitem__(key, __class__())
                self.__getitem__(key).update(value, override)
            else:
                assert override or not hasattr(
                    self, key
                ), f"override option is false. {key} exists in the original dict"
                self.__setitem__(key, value)

    def __getitem__(self, k):
        k = self._normalize_key(k)
        return super().__getitem__(k)

    def __setitem__(self, k, v) -> None:
        k = self._normalize_key(k)
        assert k not in self._reserved_keys, f"'{k}' cannot be used"
        if type(v) == dict:
            v = __class__(v)
        super().__setitem__(k, v)

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
        return obj.detach()

    def cpu(self):
        self._recursive_apply(self, self._cpu_impl)
        return self

    @staticmethod
    def _cpu_impl(obj):
        return obj.cpu()

    def to(self, target):
        self._recursive_apply(self, partial(self._to_impl, target=target))
        return self

    @staticmethod
    def _to_impl(obj, target):
        return obj.to(target)

    @classmethod
    def _recursive_apply(cls, obj, apply_fn):
        if isinstance(obj, torch.Tensor):
            obj = apply_fn(obj)
        elif isinstance(obj, (list, tuple)):
            for index in range(len(obj)):
                obj[index] = cls._recursive_apply(obj[index], apply_fn)
        elif isinstance(obj, dict):
            for key in list(obj.keys()):
                obj[key] = cls._recursive_apply(obj[key], apply_fn)
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
        return *list(selected.values()), deselected

    def select(self, *names: List[str]):
        return self.subset(*names, as_type="dict")

    def deselect(self, *names: List[str]):
        return self.subset(*names, as_type="dict", exclude=True)
