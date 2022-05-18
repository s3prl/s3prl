import copy
from unicodedata import bidirectional

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

        self.forward_lstms = nn.ModuleList(
            [
                nn.LSTM(
                    input_size=d, hidden_size=d, batch_first=True, bidirectional=False
                )
                for _ in range(self.encoder_layers)
            ]
        )
        self.backward_lstms = nn.ModuleList(
            [
                nn.LSTM(
                    input_size=d, hidden_size=d, batch_first=True, bidirectional=False
                )
                for _ in range(self.encoder_layers)
            ]
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

        forward_outputs = []
        packed_rnn_outputs = packed_rnn_inputs
        for forward_lstm in self.forward_lstms:
            packed_rnn_outputs, _ = forward_lstm(packed_rnn_outputs)
            forward_outputs.append(packed_rnn_outputs)

        packed_rnn_inputs = pack_padded_sequence(
            self.flipBatch(features, seq_lengths),
            seq_lengths,
            batch_first=True,
            enforce_sorted=False,
        )

        backward_outputs = []
        packed_rnn_outputs = packed_rnn_inputs
        for backward_lstm in self.backward_lstms:
            packed_rnn_outputs, _ = backward_lstm(packed_rnn_outputs)
            backward_outputs.append(packed_rnn_outputs)

        concat_layer_output = []
        for forward_output, backward_output in zip(forward_outputs, backward_outputs):
            x_forward, _ = pad_packed_sequence(
                forward_output, batch_first=True, total_length=max_seq_len
            )
            x_backward, _ = pad_packed_sequence(
                backward_output, batch_first=True, total_length=max_seq_len
            )
            x_backward = self.flipBatch(x_backward, seq_lengths)
            concat_layer_output.append(torch.cat((x_forward, x_backward), dim=-1))

        return concat_layer_output
