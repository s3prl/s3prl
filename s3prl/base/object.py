import logging

from s3prl.base.argument import Argument

from .checkpoint import Checkpoint

logger = logging.getLogger(__name__)


class Object(Checkpoint, Argument):
    def __init__(self) -> None:
        super().__init__()
