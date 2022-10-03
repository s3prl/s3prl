import random
from typing import List, Tuple

import librosa
import torch
import torchaudio

from . import Dataset

torchaudio.set_audio_backend("sox_io")


class LoadAudio(Dataset):
    """
    Args:
        start_secs: use None if load from start
        end_secs: use None if load to end
    """

    def __init__(
        self,
        filepaths: List[str],
        start_secs: List[float] = None,
        end_secs: List[float] = None,
        sox_effects: Tuple[Tuple[str]] = None,
        individual_sox_effects: List[Tuple[Tuple[str]]] = None,
        max_secs: float = None,
        generator: random.Random = None,
        sample_rate: int = 16000,
    ) -> None:
        super().__init__()
        self.filepaths = filepaths
        self.start_secs = start_secs
        self.end_secs = end_secs
        if generator is None:
            generator = random.Random(12345678)
        self.generator = generator

        self.sample_rate = sample_rate
        self.max_secs = max_secs

        assert int(start_secs is not None) + int(end_secs is not None) in [
            0,
            2,
        ], "start_secs and end_secs must both be given if anyone is given"

        assert (
            int(sox_effects is not None) + int(individual_sox_effects is not None) <= 1
        )
        if sox_effects is not None:
            individual_sox_effects = [sox_effects for _ in range(len(filepaths))]
        self.individual_sox_effects = individual_sox_effects

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, index: int):
        start_sec = None if self.start_secs is None else self.start_secs[index]
        start_sec = start_sec or 0.0
        end_sec = None if self.end_secs is None else self.end_secs[index]
        duration = None if end_sec is None else (self.end_secs[index] - start_sec)

        y, sr = librosa.load(
            self.filepaths[index],
            sr=self.sample_rate,
            offset=start_sec,
            duration=duration,
        )
        assert sr == self.sample_rate
        wav = torch.FloatTensor(y).view(1, -1)

        if self.individual_sox_effects is not None:
            wav, sr = torchaudio.sox_effects.apply_effects_tensor(
                wav, sr, effects=self.individual_sox_effects[index]
            )

        if sr != self.sample_rate:
            wav, sr = torchaudio.transforms.Resample(sr, self.sample_rate)(wav)

        if self.max_secs is not None:
            secs = wav.size(-1) / self.sample_rate
            if secs > self.max_secs:
                max_samples = round(self.max_secs * self.sample_rate)
                start = self.generator.randint(0, wav.size(-1) - max_samples)
                wav = wav[:, start : start + max_samples]

        wav = wav.view(-1)
        return {
            "wav_path": self.filepaths[index],
            "wav_len": len(wav),
            "wav": wav,
        }
