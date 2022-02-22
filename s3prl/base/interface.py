import re
import abc
import types
import inspect
import logging
import functools

logger = logging.getLogger(__name__)


class Interface(abc.ABC):
    def __init__(self):
        super().__init__()

    @classmethod
    def interface(cls, instance) -> None:
        pass

    @classmethod
    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

        funcname = "interface"
        func: types.FunctionType = getattr(cls, funcname)
        source = inspect.getsource(func)
        assert inspect.ismethod(func) and re.search(
            f"@classmethod", source
        ), "interface function must be @classmethod"

        if func != getattr(__class__, funcname):
            def_format = rf"def interface\(cls, instance.*\)"
            assert re.search(def_format, source), (
                f"Please follow the consistent function definition syntax: {def_format}. "
                f"Detected source:\n{source}"
            )
            assert re.search(rf"super\(\).interface\(instance\)", source), (
                f"You should call parent's interface in {func.__module__}.{func.__qualname__}. "
                f"Detected source:\n{source}"
            )

        cls_init = cls.__init__

        @functools.wraps(cls_init)
        def init_wrapper(*args, **kwargs):
            cls_init(*args, **kwargs)
            instance: type = args[0]

            if instance.__class__ == cls:
                cls.interface(instance)

        cls.__init__ = init_wrapper
