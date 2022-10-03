import copy

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# -------------#
class Decoar(nn.Module):
    def __init__(self):
        """
        input_size: an int indicating the input feature size, e.g., 80 for Mel.
        hidden_size: an int indicating the RNN hidden size.
        num_layers: an int indicating the number of RNN layers.
        dropout: a float indicating the RNN dropout rate.
        residual: a bool indicating whether to apply residual connections.
        """
        super(Decoar, self).__init__()

        self.embed = 80
        d = 1024
        self.encoder_layers = 4
        self.post_extract_proj = nn.Linear(self.embed, d)

        self.forward_lstm = nn.LSTM(
            input_size=d,
            hidden_size=d,
            num_layers=self.encoder_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.backward_lstm = nn.LSTM(
            input_size=d,
            hidden_size=d,
            num_layers=self.encoder_layers,
            batch_first=True,
            bidirectional=False,
        )

    def flipBatch(self, data, lengths):
        assert data.shape[0] == len(lengths), "Dimension Mismatch!"
        for i in range(data.shape[0]):
            data[i, : lengths[i]] = data[i, : lengths[i]].flip(dims=[0])

        return data

    def forward(self, features, padding_mask=None):
        max_seq_len = features.shape[1]
        features = self.post_extract_proj(features)

        if padding_mask is not None:
            extra = padding_mask.size(1) % features.size(1)
            if extra > 0:
                padding_mask = padding_mask[:, :-extra]
            padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
            padding_mask = padding_mask.all(-1)

        seq_lengths = (~padding_mask).sum(dim=-1).tolist()

        packed_rnn_inputs = pack_padded_sequence(
            features, seq_lengths, batch_first=True, enforce_sorted=False
        )

        packed_rnn_outputs, _ = self.forward_lstm(packed_rnn_inputs)
        x_forward, _ = pad_packed_sequence(
            packed_rnn_outputs, batch_first=True, total_length=max_seq_len
        )

        packed_rnn_inputs = pack_padded_sequence(
            self.flipBatch(features, seq_lengths),
            seq_lengths,
            batch_first=True,
            enforce_sorted=False,
        )

        packed_rnn_outputs, _ = self.backward_lstm(packed_rnn_inputs)
        x_backward, _ = pad_packed_sequence(
            packed_rnn_outputs, batch_first=True, total_length=max_seq_len
        )
        x_backward = self.flipBatch(x_backward, seq_lengths)

        return torch.cat((x_forward, x_backward), dim=-1)
