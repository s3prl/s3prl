import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim, bottleneck_dim, hidden_dim, **kwargs):
        super(Model, self).__init__()

        self.connector = nn.Linear(input_dim, bottleneck_dim)
        self.fc1 = nn.Linear(bottleneck_dim, hidden_dim)
        self.attention_linear = nn.Linear(hidden_dim, 1)

    def forward(self, features):
        # transforming
        hiddens = F.relu(self.connector(features))
        hiddens = torch.tanh(self.fc1(hiddens))

        # attentive pooling
        attention_weights = F.softmax(self.attention_linear(hiddens), dim=1)
        embeds = torch.sum(hiddens * attention_weights, dim=1)

        return embeds
