import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim, bottleneck_dim, hidden_dim, num_layers, **kwargs):
        super(Model, self).__init__()

        self.connector = nn.Linear(input_dim, bottleneck_dim)
        self.rnn = nn.LSTM(
            input_size=bottleneck_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.attention_linear = nn.Linear(hidden_dim, 1)

    def forward(self, features):
        # transforming
        hiddens = F.relu(self.connector(features))
        lstm_outputs, _ = self.rnn(hiddens)
        hiddens = torch.tanh(lstm_outputs)

        # attentive pooling
        attention_weights = F.softmax(self.attention_linear(hiddens), dim=1)
        embeds = torch.sum(hiddens * attention_weights, dim=1)

        return embeds
