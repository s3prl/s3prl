import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim, **kwargs):
        super(Model, self).__init__()

        hid_dim = kwargs["hidden_dim"]
        emb_dim = kwargs["embedding_dim"]

        self.connector = nn.Linear(input_dim, hid_dim)
        self.fc1 = nn.Linear(hid_dim, emb_dim)

    def forward(self, features):
        hiddens = F.relu(self.connector(features))
        hiddens = self.fc1(hiddens)
        embeds = hiddens.mean(dim=1)
        embeds = embeds.div(embeds.norm(2, keepdim=True))
        return embeds
