import torch
import torch.nn as nn
import torch.nn.functional as F
from downstream.model import UtteranceLevel_Linear, AP, MP


class Model(nn.Module):
    def __init__(self, input_dim, output_class_num, **kwargs):
        super(Model, self).__init__()
        hidden_dim = kwargs["hidden_dim"]
        self.connector = nn.Linear(input_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_class_num)

    def forward(self, features):
        features = F.relu(self.connector(features))
        features = self.fc1(features)
        pooled = features.mean(dim=1)
        predicted = self.fc2(pooled)
        return predicted

class UtterLinear(nn.Module):
    def __init__(self, input_dim, output_class_num, pooling_name, **kwargs):
        super(UtterLinear, self).__init__()
        self.model = UtteranceLevel_Linear(input_dim=input_dim, class_num=output_class_num)
        self.pooling = eval(pooling_name)(input_dim=input_dim)

    
    def forward(self, features, features_len):
        device = features.device
        len_masks = torch.lt(torch.arange(features_len.max()).unsqueeze(0).to(device), features_len.unsqueeze(1))
        features = self.pooling(features, len_masks)
        predicted = self.model(features)

        return predicted