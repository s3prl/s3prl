import types
import pickle
import hashlib
import inspect
import logging
import functools
from typing import Any
from pathlib import Path
from filelock import FileLock

logger = logging.getLogger(__name__)

# TODO: make this changeable
_use_cache = True
_cache_root = Path.home() / ".cache/s3prl/"
_cache_root.mkdir(exist_ok=True, parents=True)


class set_use_cache:
    def __init__(self, enable: bool):
        self.enable = enable
        self.prev = None

    def __enter__(self):
        global _use_cache
        self.prev = _use_cache
        _use_cache = self.enable

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        global _use_cache
        _use_cache = self.prev


def _string_to_filename(string):
    assert type(string) is str
    m = hashlib.sha256()
    m.update(str.encode(string))
    return str(m.hexdigest())


def cache(signatures: list = None):
    if signatures is not None:
        for sig in signatures:
            assert isinstance(sig, str)

    def cached_func(func: types.FunctionType):
        unique_name = func.__qualname__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            folder = _cache_root / unique_name
            folder.mkdir(exist_ok=True, parents=True)

            sig = inspect.signature(func)
            ba = sig.bind(*args, **kwargs)
            ba.apply_defaults()
            arguments = ba.arguments
            if signatures is not None:
                arguments = {k: v for k, v in arguments.items() if k in signatures}

            description = f"SOURCE:\n{inspect.getsource(func)}\n"
            for key, value in arguments.items():
                description += f"{key.upper()}:\n{str(value)}\n"

            filename = _string_to_filename(description)
            cache_pkl = folder / f"{filename}.pkl"
            cache_descriptor = folder / f"{filename}.txt"
            signature = f"{func.__qualname__} with arguments {arguments}"
            logger.debug(f"caching signature: {signature}")

            if cache_pkl.is_file():
                if _use_cache:
                    try:
                        with cache_pkl.open("rb") as file:
                            result = pickle.load(file)
                    except Exception:
                        logger.debug(
                            f"Cache found but cannot be loaded, Run the function {func.__qualname__}"
                        )
                    else:
                        logger.debug(
                            f"Cache found and successfully loaded. Skip running the function {func.__qualname__}"
                        )
                        return result
                else:
                    logger.debug(
                        f"Cache found but not used. Run the function {func.__qualname__}"
                    )
            else:
                logger.debug(f"Cache not found. Run the function: {func.__qualname__}")

            result = func(*args, **kwargs)

            cache_lock = folder / f"{filename}.lock"
            with FileLock(cache_lock):
                if not cache_pkl.is_file():
                    with cache_pkl.open("wb") as file:
                        pickle.dump(result, file)

                    logger.debug(f"Cache dumped for: {signature}")

                    with cache_descriptor.open("w") as file:
                        print(description, file=file)

            return result

        return wrapper

    return cached_func
