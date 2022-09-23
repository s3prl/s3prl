"""
Speaker verification loss

Authors:
  * Haibin Wu 2022
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "softmax",
    "amsoftmax",
]


class softmax(nn.Module):
    """
    The standard softmax loss in an unified interface for all speaker-related softmax losses
    """

    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self._indim = input_size
        self._outdim = output_size

        self.fc = nn.Linear(input_size, output_size)
        self.criertion = nn.CrossEntropyLoss()

    @property
    def input_size(self):
        return self._indim

    @property
    def output_size(self):
        return self._outdim

    def forward(self, x: torch.Tensor, label: torch.LongTensor):
        """
        Args:
            x (torch.Tensor): (batch_size, input_size)
            label (torch.LongTensor): (batch_size, )

        Returns:
            loss (torch.float)
            logit (torch.Tensor): (batch_size, )
        """

        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.input_size

        x = F.normalize(x, dim=1)
        x = self.fc(x)
        loss = self.criertion(x, label)

        return loss, x


class amsoftmax(nn.Module):
    """
    AMSoftmax

    Args:
        input_size (int): The input feature size
        output_size (int): The output feature size
        margin (float): Hyperparameter denotes the margin to the decision boundry
        scale (float): Hyperparameter that scales the cosine value
    """

    def __init__(
        self, input_size: int, output_size: int, margin: float = 0.2, scale: float = 30
    ):
        super().__init__()
        self._indim = input_size
        self._outdim = output_size
        self.margin = margin
        self.scale = scale

        self.W = torch.nn.Parameter(
            torch.randn(input_size, output_size), requires_grad=True
        )
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)

    @property
    def input_size(self):
        return self._indim

    @property
    def output_size(self):
        return self._outdim

    def forward(self, x: torch.Tensor, label: torch.LongTensor):
        """
        Args:
            x (torch.Tensor): (batch_size, input_size)
            label (torch.LongTensor): (batch_size, )

        Returns:
            loss (torch.float)
            logit (torch.Tensor): (batch_size, )
        """

        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.input_size

        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm)
        label_view = label.view(-1, 1)
        if label_view.is_cuda:
            label_view = label_view.cpu()
        delt_costh = torch.zeros(costh.size()).scatter_(1, label_view, self.margin)
        if x.is_cuda:
            delt_costh = delt_costh.cuda()
        costh_m = costh - delt_costh
        costh_m_s = self.scale * costh_m
        loss = self.ce(costh_m_s, label)

        return loss, costh_m_s
