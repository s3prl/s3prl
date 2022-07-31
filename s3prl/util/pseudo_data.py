import shutil
import random
import tempfile
from pathlib import Path
from typing import Any, List

import torch
import torchaudio

SAMPLE_RATE = 16000


class pseudo_audio:
    def __init__(self, secs: List[int], sample_rate: int = SAMPLE_RATE):
        self.tempdir = Path(tempfile.TemporaryDirectory().name)
        self.tempdir.mkdir(parents=True, exist_ok=True)
        self.num_samples = []
        for n, sec in enumerate(secs):
            wav = torch.randn(1, sample_rate * sec)
            torchaudio.save(
                str(self.tempdir / f"{n}.wav"), wav, sample_rate=sample_rate
            )
            self.num_samples.append(wav.size(-1))
        self.filepaths = [str(self.tempdir / f"{i}.wav") for i in range(len(secs))]

    def __enter__(self):
        return self.filepaths, self.num_samples

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        shutil.rmtree(self.tempdir)


def get_pseudo_wavs(
    seed: int = 0,
    n: int = 3,
    min_secs: int = 3,
    max_secs: int = 15,
    sample_rate: int = SAMPLE_RATE,
    device: str = "cpu",
):
    random.seed(seed)
    torch.manual_seed(seed)

    wavs = []
    wav_lengths = []
    for _ in range(n):
        wav_length = random.randint(min_secs * sample_rate, max_secs * sample_rate)
        wav = torch.randn(wav_length, requires_grad=True)

        wav_lengths.append(wav_length)
        wavs.append(wav)

    return wavs
