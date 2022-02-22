from argparse import Namespace

AVAILABLE_NAMES = [
    "output",
    "output_len",
    "hidden_states",
    "attention_mask",
    "loss",
    "label",
    "prediction",
    "wav",
    "wav_len",
]


class Output(Namespace):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            assert key in AVAILABLE_NAMES
