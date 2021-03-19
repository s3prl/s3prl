import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_dim, clipping=False, **kwargs):
        super(Model, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.clipping = lambda x: x if clipping is False else nn.Tanh()(x) * 2 + 3

    def forward(self, features, lengths):
        lengths = lengths.unsqueeze(-1).long()
        x = self.linear(features)
        x = self.clipping(x)
        frame_score = x.squeeze(-1)
        cum_score = frame_score.cumsum(-1)
        uttr_score = (cum_score.gather(-1, lengths - 1) / lengths).squeeze(-1)

        return frame_score, uttr_score
