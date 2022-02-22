import logging
from argparse import Namespace

from . import init
from .checkpoint import Checkpoint

logger = logging.getLogger(__name__)


class Object(Checkpoint):
    @init.method
    def __init__(self):
        super().__init__()
        self.arguments = None

    @classmethod
    def interface(cls, instance) -> None:
        super().interface(instance)
        assert isinstance(instance.arguments, Namespace), type(instance.arguments)
