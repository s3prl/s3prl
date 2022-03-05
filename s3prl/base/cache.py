import types
import pickle
import hashlib
import inspect
import logging
import functools
from pathlib import Path

logger = logging.getLogger(__name__)

# TODO: make this changeable
_cache_root = Path.home() / ".cache/s3prl/"
_cache_root.mkdir(exist_ok=True, parents=True)


def _string_to_filename(string):
    assert type(string) is str
    m = hashlib.sha256()
    m.update(str.encode(string))
    return str(m.hexdigest())


def cache(func: types.FunctionType):
    unique_name = func.__qualname__

    @functools.wraps(func)
    def wrapper(*args, **kwargs):

        folder = _cache_root / unique_name
        folder.mkdir(exist_ok=True, parents=True)

        sig = inspect.signature(func)
        ba = sig.bind(*args, **kwargs)
        ba.apply_defaults()
        arguments = ba.arguments

        description = f"SOURCE:\n{inspect.getsource(func)}\n"
        for key, value in arguments.items():
            description += f"{key.upper()}:\n{str(value)}\n"

        filename = _string_to_filename(description)
        cache_pkl = folder / f"{filename}.pkl"
        cache_descriptor = folder / f"{filename}.txt"
        signature = f"{func.__qualname__} with arguments {arguments}"

        if cache_pkl.is_file():
            try:
                with cache_pkl.open("rb") as file:
                    result = pickle.load(file)
            except Exception:
                pass
            else:
                logger.info(f"Cache found and loaded: {signature}")
                return result

        logger.info(f"Cache not found. Run the function: {signature}")
        result = func(*args, **kwargs)

        with cache_pkl.open("wb") as file:
            pickle.dump(result, file)
        logger.info(f"Cache dumped for: {signature}")

        with cache_descriptor.open("w") as file:
            print(description, file=file)

        return result

    return wrapper
