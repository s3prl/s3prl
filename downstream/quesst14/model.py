import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim, **kwargs):
        super(Model, self).__init__()

        btn_dim = kwargs["bottleneck_dim"]
        hid_dim = kwargs["hidden_dim"]
        num_layers = kwargs["num_layers"]

        self.connector = nn.Linear(input_dim, btn_dim)
        self.rnn = nn.LSTM(
            input_size=btn_dim,
            hidden_size=hid_dim,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, features):
        hiddens = F.relu(self.connector(features))
        outputs, _ = self.rnn(hiddens)
        embeds = outputs.mean(dim=-1)
        embeds = embeds.div(embeds.norm(2, keepdim=True))
        return embeds
