import logging
import numpy as np
from time import time
from typing import Any
from contextlib import ContextDecorator

logger = logging.getLogger(__name__)


class benchmark(ContextDecorator):
    def __init__(self, name: str, freq: int = 20) -> None:
        super().__init__()
        self.name = name
        self.freq = freq
        self.history = []
        self.n = 0

    def __enter__(self):
        self.start = time()
        self.n += 1

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        seconds = time() - self.start
        self.history.append(seconds)
        if self.n % self.freq == 0:
            logger.info(
                f"{self.name}: {seconds} secs, avg {np.array(self.history).mean()} secs"
            )
