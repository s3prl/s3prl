import torch.nn as nn

from .utils import NetworkCommonMixIn


class AudioNTT2020Task6(nn.Module, NetworkCommonMixIn):
    """DCASE2020 Task6 NTT Solution Audio Embedding Network."""

    def __init__(self, n_mels, d):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * (n_mels // (2**3)), d),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(d, d),
            nn.ReLU(),
        )
        self.d = d

    def forward(self, x):
        x = self.features(x)  # (batch, ch, mel, time)
        x = x.permute(0, 3, 2, 1)  # (batch, time, mel, ch)
        B, T, D, C = x.shape
        x = x.reshape((B, T, C * D))  # (batch, time, mel*ch)
        x = self.fc(x)
        return x


class AudioNTT2020(AudioNTT2020Task6):
    """BYOL-A General Purpose Representation Network.

    This is an extension of the DCASE 2020 Task 6 NTT Solution Audio Embedding Network.
    """

    # sample rate and embedding sizes are required model attributes for the HEAR API
    sample_rate = 16000
    embedding_size = 2048
    scene_embedding_size = embedding_size
    timestamp_embedding_size = embedding_size

    def __init__(self, n_mels=64, d=512):
        super().__init__(n_mels=n_mels, d=d)

    def forward(self, x):
        x = super().forward(x)
        x = x.mean(1) + x.amax(1)
        assert x.shape[1] == self.d and x.ndim == 2
        return x
