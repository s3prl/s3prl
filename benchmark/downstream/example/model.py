import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim, output_class_num, **kwargs):
        super(Model, self).__init__()
        self.linear = nn.Linear(input_dim, output_class_num)

    def forward(self, features):
        pooled = features.mean(dim=1)
        predicted = self.linear(pooled)
        return F.log_softmax(predicted, dim=-1)
