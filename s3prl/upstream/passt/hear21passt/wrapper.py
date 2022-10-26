import torch
import torch.nn.functional as F
from torch import nn


class PasstBasicWrapper(nn.Module):
    def __init__(
        self,
        mel: nn.Module,
        net: nn.Module,
        max_model_window=10000,
        timestamp_window=160,
        timestamp_hop=50,
        scene_hop=2500,
        scene_embedding_size=1295,
        timestamp_embedding_size=1295,
        mode="all",
    ):
        """
        @param mel: spectrogram extractor
        @param net: network module
        @param max_model_window: maximum clip length allowed by the model (milliseconds).
        @param timestamp_hop: the hop lengh for timestamp embeddings (milliseconds).
        @param scene_hop: the hop lengh for scene embeddings (milliseconds).
        @param scene_embedding_size:
        @param timestamp_embedding_size:
        @param mode: "all", "embed_only", "logits"
        """
        torch.nn.Module.__init__(self)
        self.mel = mel
        self.net = net
        self.device_proxy = nn.Parameter(torch.zeros((1)))
        self.sample_rate = mel.sr
        self.timestamp_window = int(timestamp_window * self.sample_rate / 1000)
        self.max_model_window = int(max_model_window * self.sample_rate / 1000)
        self.timestamp_hop = int(timestamp_hop * self.sample_rate / 1000)
        self.scene_hop = int(scene_hop * self.sample_rate / 1000)
        self.scene_embedding_size = scene_embedding_size
        self.timestamp_embedding_size = timestamp_embedding_size
        self.mode = mode

    def device(self):
        return self.device_proxy.device

    def forward(self, x):
        specs = self.mel(x)
        specs = specs.unsqueeze(1)
        x, features = self.net(specs)
        if self.mode == "all":
            embed = torch.cat([x, features], dim=1)
        elif self.mode == "embed_only":
            embed = features
        elif self.mode == "logits":
            embed = x
        else:
            raise RuntimeError(
                f"mode='{self.mode}' is not recognized not in: all, embed_only, logits"
            )
        return embed

    def get_scene_embeddings(self, audio):
        """
        audio: n_sounds x n_samples of mono audio in the range [-1, 1]. All sounds in a batch will be padded/trimmed to the same length.
        model: Loaded Model.
        Returns:
        embedding: A float32 Tensor with shape (n_sounds, model.scene_embedding_size).
        """
        n_sounds, n_samples = audio.shape
        if n_samples <= self.max_model_window:
            embed = self(audio.contiguous())
            return embed
        embeddings, timestamps = self.get_timestamp_embeddings(
            audio, window_size=self.max_model_window, hop=self.scene_hop
        )
        return embeddings.mean(axis=1)

    def get_timestamp_embeddings(
        self, audio: torch.Tensor, window_size=None, hop=None, pad=None
    ):
        """
        audio: n_sounds x n_samples of mono audio in the range [-1, 1]. All sounds in a batch will be padded/trimmed to the same length.
        model: Loaded Model.
        Returns:
        embedding: A float32 Tensor with shape (n_sounds, n_timestamps, model.timestamp_embedding_size).
        timestamps: A float32 Tensor with shape (`n_sounds, n_timestamps). Centered timestamps in milliseconds corresponding to each embedding in the output.
        """
        if hop is None:
            hop = self.timestamp_hop
        if window_size is None:
            window_size = self.timestamp_window
        if pad is None:
            pad = window_size // 2

        n_sounds, n_samples = audio.shape
        audio = audio.unsqueeze(1)  # n_sounds,1, (n_samples+pad*2)
        # print(audio.shape)
        padded = F.pad(audio, (pad, pad), mode="reflect")
        # print(padded.shape)
        padded = padded.unsqueeze(1)  # n_sounds,1, (n_samples+pad*2)
        # print(padded.shape)
        segments = (
            F.unfold(padded, kernel_size=(1, window_size), stride=(1, hop))
            .transpose(-1, -2)
            .transpose(0, 1)
        )
        timestamps = []
        embeddings = []
        for i, segment in enumerate(segments):
            timestamps.append(i)
            emb = self(segment)
            embeddings.append(emb)
        timestamps = torch.as_tensor(timestamps) * hop * 1000.0 / self.sample_rate

        embeddings = torch.stack(embeddings).transpose(
            0, 1
        )  # now n_sounds, n_timestamps, timestamp_embedding_size
        timestamps = timestamps.unsqueeze(0).expand(n_sounds, -1)

        return embeddings, timestamps

    def get_timestamp_mels(
        self, audio: torch.Tensor, window_size=None, hop=None, pad=None
    ):
        """
        audio: n_sounds x n_samples of mono audio in the range [-1, 1]. All sounds in a batch will be padded/trimmed to the same length.
        model: Loaded Model.
        Returns:
        embedding: A float32 Tensor with shape (n_sounds, n_timestamps, model.timestamp_embedding_size).
        timestamps: A float32 Tensor with shape (`n_sounds, n_timestamps). Centered timestamps in milliseconds corresponding to each embedding in the output.
        """
        if hop is None:
            hop = self.timestamp_hop
        if window_size is None:
            window_size = self.timestamp_window
        if pad is None:
            pad = window_size // 2

        n_sounds, n_samples = audio.shape
        audio = audio.unsqueeze(1)  # n_sounds,1, (n_samples+pad*2)
        # print(audio.shape)
        padded = F.pad(audio, (pad, pad), mode="reflect")
        # print(padded.shape)
        padded = padded.unsqueeze(1)  # n_sounds,1, (n_samples+pad*2)
        # print(padded.shape)
        segments = (
            F.unfold(padded, kernel_size=(1, window_size), stride=(1, hop))
            .transpose(-1, -2)
            .transpose(0, 1)
        )
        timestamps = []
        embeddings = []
        for i, segment in enumerate(segments):
            timestamps.append(i)
            embeddings.append(self.mel(segment).reshape(n_sounds, 128 * 6))
        timestamps = torch.as_tensor(timestamps) * hop * 1000.0 / self.sample_rate

        embeddings = torch.stack(embeddings).transpose(
            0, 1
        )  # now n_sounds, n_timestamps, timestamp_embedding_size
        timestamps = timestamps.unsqueeze(0).expand(n_sounds, -1)

        return embeddings, timestamps
