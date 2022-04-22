import logging
from time import time
from typing import Any
from contextlib import ContextDecorator

logger = logging.getLogger(__name__)

class benchmark(ContextDecorator):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def __enter__(self):
        self.start = time()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        logger.info(f"{self.name}: {time() - self.start}")
