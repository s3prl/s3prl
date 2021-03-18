import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_dim, **kwargs):
        super(Model, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, features):
        x = self.linear(features)
        frame_score = x.squeeze(-1)
        uttr_score = frame_score.mean(dim=-1)

        return frame_score, uttr_score
