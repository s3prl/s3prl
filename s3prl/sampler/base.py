from typing import Iterator, Optional, TypeVar

from torch.utils.data import Sampler as _Sampler

from s3prl import Object
from s3prl.util import registry

T_co = TypeVar("T_co", covariant=True)


class Sampler(_Sampler[T_co]):
    def __init_subclass__(cls) -> None:
        registry.put(cls.__name__)(cls)
        return super().__init_subclass__()
