import abc

import torch
import torchaudio

from s3prl import Object, Output, init


class Loader(Object):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def __call__(self, source):
        raise NotImplementedError


class PseudoLoader(Loader):
    def __init__(self):
        super().__init__()

    def __call__(self, source):
        return Output(output=torch.randn(160000, 1))


class TorchaudioLoader(Loader):
    def __init__(
        self, sample_rate: int = 16000, torch_audio_backend: str = "sox_io"
    ) -> None:
        assert isinstance(sample_rate, int)
        self.sample_rate = sample_rate
        self.torch_audio_backend = torch_audio_backend

    def __call__(self, source):
        if self.torch_audio_backend is not None:
            torchaudio.set_audio_backend(self.torch_audio_backend)
        wav, sr = torchaudio.load(source)
        assert self.sample_rate == sr
        wav = wav.view(-1, 1)
        return Output(output=wav)


class TorchaudioMetadataLoader(Loader):
    def __init__(
        self, sample_rate: int = 16000, torch_audio_backend: str = "sox_io"
    ) -> None:
        assert isinstance(sample_rate, int)
        assert torch_audio_backend == "sox_io"

        self.sample_rate = sample_rate
        self.torch_audio_backend = torch_audio_backend

    def __call__(self, source):
        if self.torch_audio_backend is not None:
            torchaudio.set_audio_backend(self.torch_audio_backend)
        info = torchaudio.info(source)
        assert self.sample_rate == info.sample_rate
        return Output(timestamp=info.num_frames)
