import torch
import torch.nn as nn


class MosDownstream(nn.Module):
    def __init__(self, upstream_dim, projector_dim, clipping, attention_pooling):
        super(MosDownstream, self).__init__()
        self.connector = nn.Linear(upstream_dim, projector_dim)
        self.model = MosDownstreamModule(
            input_dim=projector_dim,
            clipping=clipping,
            attention_pooling=attention_pooling,
        )

    def forward(self, features):
        features = self.connector(features)
        scores = self.model(features)
        return scores


class MosDownstreamModule(nn.Module):
    def __init__(
        self,
        input_dim,
        clipping=False,
        attention_pooling=False,
        num_judges=5000,
        **kwargs
    ):
        super(MosDownstreamModule, self).__init__()
        self.mean_net_linear = nn.Linear(input_dim, 1)
        self.mean_net_clipping = clipping
        self.mean_net_pooling = (
            SelfAttentionPooling(input_dim) if attention_pooling else None
        )
        self.bias_net_linear = nn.Linear(input_dim, 1)
        self.bias_net_pooling = (
            SelfAttentionPooling(input_dim) if attention_pooling else None
        )
        self.judge_embbeding = nn.Embedding(
            num_embeddings=num_judges, embedding_dim=input_dim
        )

    def forward(self, features, judge_ids=None):
        if self.mean_net_pooling is not None:
            x = self.mean_net_pooling(features)
            segment_score = self.mean_net_linear(x)
        else:
            x = self.mean_net_linear(features)
            segment_score = x.squeeze(-1).mean(dim=-1)

        if self.mean_net_clipping:
            segment_score = torch.tanh(segment_score) * 2 + 3

        if judge_ids is None:
            return segment_score.squeeze(-1)

        else:
            time = features.shape[1]
            judge_features = self.judge_embbeding(judge_ids)
            judge_features = torch.stack([judge_features for i in range(time)], dim=1)
            bias_features = features + judge_features

            if self.bias_net_pooling is not None:
                y = self.bias_net_pooling(bias_features)
                bias_score = self.bias_net_linear(y)
            else:
                y = self.bias_net_linear(bias_features)
                bias_score = y.squeeze(-1).mean(dim=-1)
            bias_score = bias_score + segment_score

        return segment_score.squeeze(-1), bias_score.squeeze(-1)


class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """

    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)

    def forward(self, batch_rep):
        """
        input:
            batch_rep : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension

        attention_weight:
            att_w : size (N, T, 1)

        return:
            utter_rep: size (N, H)
        """
        softmax = nn.functional.softmax
        att_w = softmax(self.W(batch_rep).squeeze(-1)).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep
