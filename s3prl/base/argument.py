import functools
import inspect
import types
from argparse import Namespace


def save_arguments(func: types.FunctionType):
    @functools.wraps(func)
    def arguments_saved_init(self, *args, **kwargs):
        sig = inspect.signature(func)

        params = list(sig.parameters.values())
        self_name = params[0].name

        ba = sig.bind(self, *args, **kwargs)
        ba.apply_defaults()
        arguments = ba.arguments
        arguments.pop(self_name)

        if not hasattr(self, "_arguments"):
            self._arguments = Namespace()
        assert isinstance(self._arguments, Namespace)
        for key, value in arguments.items():
            setattr(self._arguments, key, value)

        func(self, *args, **kwargs)

    return arguments_saved_init


class Argument:
    @property
    def arguments(self):
        return self._arguments

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls.__init__ = save_arguments(cls.__init__)
