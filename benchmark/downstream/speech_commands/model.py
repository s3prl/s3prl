import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim, output_class_num, **kwargs):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_dim, kwargs["hidden_dim"])
        self.fc2 = nn.Linear(kwargs["hidden_dim"], output_class_num)

    def forward(self, features):
        pooled = features.mean(dim=1)
        predicted = self.fc2(F.relu(self.fc1(pooled)))
        return predicted
