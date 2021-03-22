import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """

    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)

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
        weights = F.softmax(logits + masks).unsqueeze(-1)
        pooled_features = (features * weights).sum(dim=1)

        return pooled_features


class Model(nn.Module):
    def __init__(self, input_dim, clipping=False, attention_pooling=False, **kwargs):
        super(Model, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.clipping = clipping
        self.pooling = SelfAttentionPooling(input_dim) if attention_pooling else None

    def forward(self, features, lengths):
        """Forward a batch of data through model.

        Args:
            features: (batch_size, padded_length, bottleneck_dim)
            lengths: (batch_size,)

        Returns:
            frame_scores: (batch_size, padded_length)
            uttr_score: (batch_size, 1)
        """
        lengths = lengths.unsqueeze(-1).long()  # (batch_size, 1)
        frame_scores = self.linear(features).squeeze(-1)  # (batch_size, padded_length)

        if self.clipping:
            frame_scores = torch.tanh(frame_scores) * 2 + 3

        if self.pooling is not None:
            uttr_features = self.pooling(features, lengths)
            uttr_score = self.linear(uttr_features)
            if self.clipping:
                uttr_score = torch.tanh(uttr_score) * 2 + 3
        else:
            cum_score = frame_scores.cumsum(-1)
            uttr_score = cum_score.gather(-1, lengths - 1) / lengths

        return frame_scores, uttr_score
