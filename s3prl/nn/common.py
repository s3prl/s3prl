"""
Common probing models

Authors:
  * Leo 2022
"""

from typing import List

import torch.nn as nn

import s3prl.nn.pooling as pooling

__all__ = [
    "FrameLevel",
    "UtteranceLevel",
]


class FrameLevel(nn.Module):
    """
    The common frame-to-frame probing model

    Args:
        input_size (int): input size
        output_size (int): output size
        hidden_sizes (List[int]): a list of hidden layers' hidden size.
            by default is [256] to project all different input sizes to the same dimension.
            set empty list to use the vanilla single layer linear model
        activation_type (str): the activation class name in :obj:`torch.nn`. Set None to
            disable activation and the model is pure linear. Default: None
        activation_conf (dict): the arguments for initializing the activation class.
            Default: empty dict
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: List[int] = None,
        activation_type: str = None,
        activation_conf: dict = None,
    ):
        super().__init__()
        self._indim = input_size
        self._outdim = output_size
        hidden_sizes = hidden_sizes or [256]

        latest_size = input_size
        hidden_layers = []
        if len(hidden_sizes) > 0:
            for size in hidden_sizes:
                hidden_layers.append(nn.Linear(latest_size, size))
                if activation_type is not None:
                    hidden_layers.append(
                        getattr(nn, activation_type)(**(activation_conf or {}))
                    )
                latest_size = size

        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.final_proj = nn.Linear(latest_size, output_size)

    @property
    def input_size(self) -> int:
        return self._indim

    @property
    def output_size(self) -> int:
        return self._outdim

    def forward(self, x, x_len):
        """
        Args:
            x (torch.FloatTensor): (batch_size, seq_len, input_size)
            x_len (torch.LongTensor): (batch_size, )

        Returns:
            tuple

            1. ys (torch.FloatTensor): (batch_size, seq_len, output_size)
            2. ys_len (torch.LongTensor): (batch_size, )
        """
        ys = self.hidden_layers(x)
        ys = self.final_proj(ys)
        return ys, x_len


class UtteranceLevel(nn.Module):
    """
    Args:
        input_size (int): input_size
        output_size (int): output_size
        hidden_sizes (List[int]): a list of hidden layers' hidden size.
            by default is [256] to project all different input sizes to the same dimension.
            set empty list to use the vanilla single layer linear model
        activation_type (str): the activation class name in :obj:`torch.nn`. Set None to
            disable activation and the model is pure linear. Default: None
        activation_conf (dict): the arguments for initializing the activation class.
            Default: empty dict
        pooling_type (str): the pooling class name in :obj:`s3prl.nn.pooling`. Default: MeanPooling
        pooling_conf (dict): the arguments for initializing the pooling class.
            Default: empty dict
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: List[int] = None,
        activation_type: str = None,
        activation_conf: dict = None,
        pooling_type: str = "MeanPooling",
        pooling_conf: dict = None,
    ):
        super().__init__()
        self._indim = input_size
        self._outdim = output_size
        hidden_sizes = hidden_sizes or [256]

        latest_size = input_size
        hidden_layers = []
        if len(hidden_sizes) > 0:
            for size in hidden_sizes:
                hidden_layers.append(nn.Linear(latest_size, size))
                if activation_type is not None:
                    hidden_layers.append(
                        getattr(nn, activation_type)(**(activation_conf or {}))
                    )
                latest_size = size

        self.hidden_layers = nn.Sequential(*hidden_layers)

        pooling_conf = pooling_conf or {}
        self.pooling = getattr(pooling, pooling_type)(latest_size, **pooling_conf)
        latest_size = self.pooling.output_size
        self.final_proj = nn.Linear(latest_size, output_size)

    @property
    def input_size(self) -> int:
        return self._indim

    @property
    def output_size(self) -> int:
        return self._outdim

    def forward(self, x, x_len):
        """
        Args:
            x (torch.FloatTensor): (batch_size, seq_len, input_size)
            x_len (torch.LongTensor): (batch_size, )

        Returns:
            torch.FloatTensor

            (batch_size, output_size)
        """
        x = self.hidden_layers(x)
        x_pooled = self.pooling(x, x_len)
        y = self.final_proj(x_pooled)
        return y
