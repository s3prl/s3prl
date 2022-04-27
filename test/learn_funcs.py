import functools
import importlib
import inspect


def get_args(fn):
    @functools.wraps(fn)
    def tmp(*args, **kwargs):
        print("in wrap", fn)
        return fn(*args, **kwargs)

    return tmp


class C:
    @get_args
    def __init__(self, a) -> None:
        super().__init__()
        pass

    @get_args
    def test1(self, a) -> None:
        pass

    @classmethod
    @get_args
    def test2(cls, a) -> None:
        pass

    @staticmethod
    @get_args
    def test3(a) -> None:
        pass


module = importlib.import_module("__main__")
print("C.__init__", getattr(getattr(module, "C"), "__init__"))
print("C.test2", getattr(getattr(module, "C"), "test2"))

c = C(3)
c.test2(3)
print("c.test2", c.test2)

C.test2(3)
print("C.test2", C.test2)

c.test1(3)
print("C.test1", c.test1)

c.test3(3)
print("C.test3", c.test3)
