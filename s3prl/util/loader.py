import torch
import torchaudio

from s3prl import Object, init


class Loader(Object):
    @init.method
    def __init__(self):
        super().__init__()

    def load(self, source):
        return torch.randn(160000)


class TorchaudioLoader(Loader):
    @init.method
    def __init__(
        self, sample_rate: int = None, torch_audio_backend: str = None
    ) -> None:
        self.sample_rate = sample_rate
        self.torch_audio_backend = torch_audio_backend

    def load(self, source):
        if self.torch_audio_backend is not None:
            torchaudio.set_audio_backend(self.torch_audio_backend)
        wav, sr = torchaudio.load(source)
        if self.sample_rate is not None:
            assert self.sample_rate == sr
        return wav
