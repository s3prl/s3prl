"""
Common linear models

Authors:
  * Leo 2022
"""

from .common import FrameLevel, UtteranceLevel

__all__ = [
    "FrameLevelLinear",
    "MeanPoolingLinear",
]


class FrameLevelLinear(FrameLevel):
    """
    The frame-level linear probing model used in SUPERB Benchmark
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 256,
    ):
        super().__init__(input_size, output_size, hidden_sizes=[hidden_size])


class MeanPoolingLinear(UtteranceLevel):
    """
    The utterance-level linear probing model used in SUPERB Benchmark
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 256,
    ):
        super().__init__(input_size, output_size, hidden_sizes=[hidden_size])
