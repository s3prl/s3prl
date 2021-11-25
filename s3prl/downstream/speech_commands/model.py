import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Note that this model file was left here due to the legacy reason and is not used in the
SUPERB Benchmark. In SUPERB's speech classification tasks we use linear models, including:

- KS: speech commands
- IC: fluent speech commands
- ER: IEMOCAP emotion classification
- SID: VoxCeleb1 speaker classification

One can trace the following files:

- downstream/speech_commands/config.yaml: downstream_expert.modelrc.select=UtteranceLevel
- downstream/model.py: UtteranceLevel

This "UtteranceLevel" module is used across KS, ER, IC and SID in SUPERB, which first
linearly projects upstream's feature dimension to the same dimension (256), and then
linearly projected to the class number. Hence, it does not contain non-linearity.
"""

class Model(nn.Module):
    """
    Not used in SUPERB Benchmark
    """

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
