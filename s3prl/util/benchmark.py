"""
Benchmark the timing a block of code

Authors
  * Leo 2022
"""

import logging
from collections import defaultdict
from contextlib import ContextDecorator
from time import time
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)
_history = defaultdict(list)

__all__ = ["benchmark"]


class benchmark(ContextDecorator):
    def __init__(self, name: str, freq: int = 20) -> None:
        super().__init__()
        self.name = name
        self.freq = freq

    def __enter__(self):
        torch.cuda.synchronize()
        self.start = time()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        torch.cuda.synchronize()
        seconds = time() - self.start

        global _history
        _history[self.name].append(seconds)
        if len(_history[self.name]) % self.freq == 0:
            logger.warning(
                f"{self.name}: {seconds} secs, avg {np.array(_history[self.name]).mean()} secs"
            )
