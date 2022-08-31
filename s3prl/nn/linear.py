from .common import FrameLevel, UtteranceLevel


class FrameLevelLinear(FrameLevel):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 256,
    ):
        super().__init__(input_size, output_size, hidden_sizes=[hidden_size])


class MeanPoolingLinear(UtteranceLevel):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 256,
    ):
        super().__init__(input_size, output_size, hidden_sizes=[hidden_size])
