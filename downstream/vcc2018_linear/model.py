import torch
import torch.nn as nn


class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """

    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)
        self.softmax = nn.functional.softmax

    def forward(self, features, lengths):
        """
        input:
            features : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension
            lengths : size (N, 1), N: batch size

        attention_weight:
            weights : size (N, T, 1)

        return:
            pooled_features: size (N, H)
        """
        device = features.device
        logits = self.W(features).squeeze(-1)
        masks = (
            torch.arange(logits.size(-1)).expand_as(logits).to(device) >= lengths
        ).float()
        masks[masks.bool()] = float("-inf")
        weights = self.softmax(logits + masks).unsqueeze(-1)
        pooled_features = (features * weights).sum(dim=1)

        return pooled_features


class Model(nn.Module):
    def __init__(self, input_dim, clipping=False, attention_pooling=False, **kwargs):
        super(Model, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.clipping = lambda x: x if clipping is False else nn.Tanh()(x) * 2 + 3
        self.pooling = SelfAttentionPooling(input_dim) if attention_pooling else None

    def forward(self, features, lengths):
        lengths = lengths.unsqueeze(-1).long()
        x = self.linear(features)
        x = self.clipping(x)
        frame_score = x.squeeze(-1)
        if self.pooling is not None:
            uttr_features = self.pooling(features, lengths)
            uttr_score = self.linear(uttr_features)
            uttr_score = self.clipping(uttr_score).squeeze(-1)
        else:
            cum_score = frame_score.cumsum(-1)
            uttr_score = (cum_score.gather(-1, lengths - 1) / lengths).squeeze(-1)

        return frame_score, uttr_score
