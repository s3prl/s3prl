import math

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

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

    def forward(self, batch_rep, att_mask=None):
        """
            N: batch size, T: sequence length, H: Hidden dimension
            input:
                batch_rep : size (N, T, H)
            attention_weight:
                att_w : size (N, T, 1)
            return:
                utter_rep: size (N, H)
        """
        att_logits = self.W(batch_rep).squeeze(-1)
        if att_mask is not None:
            att_logits = att_mask + att_logits
        att_w = self.softmax(att_logits, dim=-1).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep


class CNNSelfAttention(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        kernel_size,
        padding,
        pooling,
        dropout,
        output_class_num,
        **kwargs
    ):
        super(CNNSelfAttention, self).__init__()
        self.model_seq = nn.Sequential(
            nn.AvgPool1d(kernel_size, pooling, padding),
            nn.Dropout(p=dropout),
            nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding),
        )
        self.pooling = SelfAttentionPooling(hidden_dim)
        self.out_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_class_num),
        )

    def forward(self, features, att_mask):
        features = features.transpose(1, 2)
        features = self.model_seq(features)
        out = features.transpose(1, 2)
        out = self.pooling(out, att_mask).squeeze(-1)
        predicted = self.out_layer(out)
        return predicted


class FCN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        kernel_size,
        padding,
        pooling,
        dropout,
        output_class_num,
        **kwargs,
    ):
        super(FCN, self).__init__()
        self.model_seq = nn.Sequential(
            nn.Conv1d(input_dim, 96, 11, stride=4, padding=5),
            nn.LocalResponseNorm(96),
            nn.ReLU(),
            nn.MaxPool1d(3, 2),
            nn.Dropout(p=dropout),
            nn.Conv1d(96, 256, 5, padding=2),
            nn.LocalResponseNorm(256),
            nn.ReLU(),
            nn.MaxPool1d(3, 2),
            nn.Dropout(p=dropout),
            nn.Conv1d(256, 384, 3, padding=1),
            nn.LocalResponseNorm(384),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Conv1d(384, 384, 3, padding=1),
            nn.LocalResponseNorm(384),
            nn.ReLU(),
            nn.Conv1d(384, 256, 3, padding=1),
            nn.LocalResponseNorm(256),
            nn.MaxPool1d(3, 2),
        )
        self.pooling = SelfAttentionPooling(256)
        self.out_layer = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_class_num),
        )

    def forward(self, features, att_mask):
        features = features.transpose(1, 2)
        features = self.model_seq(features)
        out = features.transpose(1, 2)
        out = self.pooling(out).squeeze(-1)
        predicted = self.out_layer(out)
        return predicted


class DeepNet(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        kernel_size,
        padding,
        pooling,
        dropout,
        output_class_num,
        **kwargs,
    ):
        super(DeepNet, self).__init__()
        self.model_seq = nn.Sequential(
            nn.Conv1d(input_dim, 10, 9),
            nn.ReLU(),
            nn.Conv1d(10, 10, 5),
            nn.ReLU(),
            nn.Conv1d(10, 10, 3),
            nn.MaxPool1d(3, 1),
            nn.BatchNorm1d(10, affine=False),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Conv1d(10, 40, 3),
            nn.ReLU(),
            nn.Conv1d(40, 40, 3),
            nn.MaxPool1d(2, 1),
            nn.BatchNorm1d(40, affine=False),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Conv1d(40, 80, 10),
            nn.ReLU(),
            nn.Conv1d(80, 80, 1),
            nn.MaxPool1d(2, 1),
            nn.BatchNorm1d(80, affine=False),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Conv1d(80, 80, 1),
        )
        self.pooling = SelfAttentionPooling(80)
        self.out_layer = nn.Sequential(
            nn.Linear(80, 30),
            nn.ReLU(),
            nn.Linear(30, output_class_num),
        )

    def forward(self, features, att_mask):
        features = features.transpose(1, 2)
        features = self.model_seq(features)
        out = features.transpose(1, 2)
        out = self.pooling(out).squeeze(-1)
        predicted = self.out_layer(out)
        return predicted


class DeepModel(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        model_type,
        pooling,
        **kwargs
    ):
        super(DeepModel, self).__init__()
        self.pooling = pooling
        self.model = eval(model_type)(input_dim=input_dim, output_class_num=output_dim, pooling=pooling, **kwargs)

    def forward(self, features, features_len):
        attention_mask = [
            torch.ones(math.ceil((l / self.pooling)))
            for l in features_len
        ]
        attention_mask = pad_sequence(attention_mask, batch_first=True)
        attention_mask = (1.0 - attention_mask) * -100000.0
        attention_mask = attention_mask.to(features.device)
        predicted = self.model(features, attention_mask)
        return predicted, None
