import types
import pickle
import hashlib
import inspect
from pathlib import Path

# TODO: make this changeable
_cache_root = Path.home() / ".cache/s3prl/"
_cache_root.mkdir(exist_ok=True)


def _string_to_filename(string):
    assert type(string) is str
    m = hashlib.sha256()
    m.update(str.encode(string))
    return str(m.hexdigest())

_registered_cache_name = []

def cache(unique_name: str = None):
    def decorator(function: types.FunctionType):
        def wrapper(*args, **kwargs):
            if unique_name is None:
                sig = inspect.signature(function)
                arguments = sig.bind(*args, **kwargs).arguments
                pattern = f"{function.__module__}.{function.__qualname__}\n"
                for key, value in arguments.items():
                    pattern += f"{key.upper()}\n{str(value)}\n"
                filename = _string_to_filename(pattern)
            else:
                assert unique_name not in _registered_cache_name, (
                    f"Duplicated cache name: {unique_name} at {function.__module__}.{function.__qualname__}"
                )
                _registered_cache_name.append(unique_name)
                filename = unique_name
            cache_path = _cache_root / f"{filename}.pkl"

            if cache_path.is_file():
                with cache_path.open("wb") as file:
                    result = pickle.load(file)
                return result

            result = function(*args, **kwargs)

            with cache_path.open("wb") as file:
                pickle.dump(result, file)

            return result
        return wrapper
    return decorator
