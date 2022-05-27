import logging

from s3prl.base.argument import Argument
from s3prl.util import registry

from .checkpoint import Checkpoint

logger = logging.getLogger(__name__)


class Object(Checkpoint, Argument):
    def __init__(self) -> None:
        super().__init__()

    def __init_subclass__(cls) -> None:
        registry.put(cls.__name__)(cls)
        return super().__init_subclass__()
