"""
RNN models used in Superb Benchmark

Authors:
  * Heng-Jui Chang 2022
  * Leo 2022
"""

from typing import List, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from s3prl.nn.interface import AbsFrameModel

__all__ = ["RNNEncoder", "SuperbDiarizationModel", "RNNLayer"]


def downsample(
    x: torch.Tensor, x_len: torch.LongTensor, sample_rate: int, sample_style: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Downsamples a sequence.

    Args:
        x (torch.Tensor): Sequence (batch, timestamps, hidden_size)
        x_len (torch.LongTensor): Sequence length (batch, )
        sample_rate (int): Downsample rate (must be greater than one)
        sample_style (str): Downsample style ("drop" or "concat")

    Raises:
        NotImplementedError: Sample style not supported.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            x (torch.Tensor): (batch, timestamps // sample_rate, output_size)
            x_len (torch.LongTensor): (batch, )
    """

    B, T, D = x.shape
    x_len = torch.div(x_len, sample_rate, rounding_mode="floor")

    if sample_style == "drop":
        # Drop the unselected timesteps
        x = x[:, ::sample_rate, :].contiguous()
    elif sample_style == "concat":
        # Drop the redundant frames and concat the rest according to sample rate
        if T % sample_rate != 0:
            x = x[:, : -(T % sample_rate), :]
        x = x.contiguous().view(B, int(T / sample_rate), D * sample_rate)
    else:
        raise NotImplementedError(f"Sample style={sample_style} not supported.")

    return x, x_len


class RNNLayer(nn.Module):
    """RNN Layer

    Args:
        input_size (int): Input size.
        hidden_size (int): Hidden size.
        module (str): RNN module (RNN, GRU, LSTM)
        dropout (float, optional): Dropout rate. Defaults to 0.0.
        bidirectional (bool, optional): Bidirectional. Defaults to False.
        proj (bool, optional): Projection layer. Defaults to False.
        layer_norm (bool, optional): Layer normalization. Defaults to False.
        sample_rate (int, optional): Downsampling rate. Defaults to 1.
        sample_style (str, optional): Downsampling style (**drop** or **concat**). Defaults to "drop".
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        module: str,
        dropout: float = 0.0,
        bidirectional: bool = False,
        proj: bool = False,
        layer_norm: bool = False,
        sample_rate: int = 1,
        sample_style: str = "drop",
    ):

        super().__init__()
        self._insize = input_size

        self.out_size = (
            hidden_size
            * (2 if bidirectional else 1)
            * (2 if sample_style == "concat" and sample_rate > 1 else 1)
        )
        self.dropout = dropout
        self.proj = proj
        self.layer_norm = layer_norm
        self.sample_rate = sample_rate
        self.sample_style = sample_style

        assert module.upper() in {"RNN", "GRU", "LSTM"}
        assert sample_style in {"drop", "concat"}

        self.layer = getattr(nn, module.upper())(
            input_size,
            hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )

        if self.layer_norm:
            rnn_out_size = hidden_size * (2 if bidirectional else 1)
            self.ln_layer = nn.LayerNorm(rnn_out_size)

        if self.dropout > 0:
            self.dp_layer = nn.Dropout(self.dropout)

        if self.proj:
            self.pj_layer = nn.Linear(self.out_size, self.out_size)

    def forward(self, xs: torch.Tensor, xs_len: torch.LongTensor):
        """
        Args:
            xs (torch.FloatTensor): (batch_size, seq_len, input_size)
            xs_len (torch.LongTensor): (batch_size, )

        Returns:
            tuple:

            1. ys (torch.FloatTensor): (batch_size, seq_len, output_size)
            2. ys_len (torch.LongTensor): (batch_size, )
        """
        if not self.training:
            self.layer.flatten_parameters()

        xs = pack_padded_sequence(
            xs, xs_len.cpu(), batch_first=True, enforce_sorted=False
        )
        output, _ = self.layer(xs)
        output, _ = pad_packed_sequence(output, batch_first=True)

        # Normalization
        if self.layer_norm:
            output = self.ln_layer(output)

        if self.dropout > 0:
            output = self.dp_layer(output)

        # Downsampling
        if self.sample_rate > 1:
            output, xs_len = downsample(
                output, xs_len, self.sample_rate, self.sample_style
            )

        # Projection
        if self.proj:
            output = torch.tanh(self.pj_layer(output))

        return output, xs_len

    @property
    def input_size(self) -> int:
        return self._insize

    @property
    def output_size(self) -> int:
        return self.out_size


class RNNEncoder(AbsFrameModel):
    """RNN Encoder for sequence to sequence modeling, e.g., ASR.

    Args:
        input_size (int): Input size.
        output_size (int): Output size.
        module (str, optional): RNN module type. Defaults to "LSTM".
        hidden_size (List[int], optional): Hidden sizes for each layer. Defaults to [1024].
        dropout (List[float], optional): Dropout rates for each layer. Defaults to [0.0].
        layer_norm (List[bool], optional): Whether to use layer norm for each layer. Defaults to [False].
        proj (List[bool], optional): Whether to use projection for each layer. Defaults to [True].
        sample_rate (List[int], optional): Downsample rates for each layer. Defaults to [1].
        sample_style (str, optional): Downsample style ("drop" or "concat"). Defaults to "drop".
        bidirectional (bool, optional): Whether RNN layers are bidirectional. Defaults to False.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        module: str = "LSTM",
        proj_size: int = 1024,
        hidden_size: List[int] = [1024],
        dropout: List[float] = [0.0],
        layer_norm: List[bool] = [False],
        proj: List[bool] = [True],
        sample_rate: List[int] = [1],
        sample_style: str = "drop",
        bidirectional: bool = False,
    ):
        super().__init__()
        self._input_size = input_size
        self._output_size = output_size

        prev_size = input_size

        self.proj = nn.Linear(prev_size, proj_size)
        prev_size = proj_size

        self.rnns = nn.ModuleList()
        for i in range(len(hidden_size)):
            rnn_layer = RNNLayer(
                input_size=prev_size,
                hidden_size=hidden_size[i],
                module=module,
                dropout=dropout[i],
                bidirectional=bidirectional,
                proj=proj[i],
                layer_norm=layer_norm[i],
                sample_rate=sample_rate[i],
                sample_style=sample_style,
            )
            self.rnns.append(rnn_layer)
            prev_size = rnn_layer.output_size

        self.linear = nn.Linear(prev_size, output_size)

    def forward(self, x: torch.Tensor, x_len: torch.LongTensor):
        """
        Args:
            xs (torch.FloatTensor): (batch_size, seq_len, input_size)
            xs_len (torch.LongTensor): (batch_size, )

        Returns:
            tuple:

            1. ys (torch.FloatTensor): (batch_size, seq_len, output_size)
            2. ys_len (torch.LongTensor): (batch_size, )
        """

        xs, xs_len = x, x_len
        xs = self.proj(xs)

        for rnn in self.rnns:
            xs, xs_len = rnn(xs, xs_len)

        logits = self.linear(xs)

        return logits, xs_len

    @property
    def input_size(self) -> int:
        return self._input_size

    @property
    def output_size(self) -> int:
        return self._output_size


class SuperbDiarizationModel(AbsFrameModel):
    """
    The exact RNN model used in SUPERB Benchmark for Speaker Diarization

    Args:
        input_size (int): input_size
        output_size (int): output_size
        rnn_layers (int): number of rnn layers
        hidden_size (int): the hidden size across all rnn layers
    """

    def __init__(
        self, input_size: int, output_size: int, rnn_layers: int, hidden_size: int
    ):
        super().__init__()
        self._input_size = input_size
        self._output_size = output_size

        self.use_rnn = rnn_layers > 0
        if self.use_rnn:
            self.rnn = nn.LSTM(
                input_size, hidden_size, num_layers=rnn_layers, batch_first=True
            )
            self.linear = nn.Linear(hidden_size, output_size)
        else:
            self.linear = nn.Linear(input_size, output_size)

    @property
    def input_size(self) -> int:
        return self._input_size

    @property
    def output_size(self) -> int:
        return self._output_size

    def forward(self, xs, xs_len):
        """
        Args:
            xs (torch.FloatTensor): (batch_size, seq_len, input_size)
            xs_len (torch.LongTensor): (batch_size, )

        Returns:
            tuple:

            1. ys (torch.FloatTensor): (batch_size, seq_len, output_size)
            2. ys_len (torch.LongTensor): (batch_size, )
        """
        features, features_len = xs, xs_len
        features = features.float()
        if self.use_rnn:
            hidden, _ = self.rnn(features)
            predicted = self.linear(hidden)
        else:
            predicted = self.linear(features)

        return predicted, features_len
