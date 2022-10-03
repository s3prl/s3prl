"""
The probing model following Hear Benchmark

Authors:
  * Hear Team 2021
  * Leo 2022
"""


from typing import List

import torch

import s3prl.nn.pooling as pooling

__all__ = ["HearFullyConnectedPrediction"]


class HearFullyConnectedPrediction(torch.nn.Module):
    """
    The specific prediction head used in the Hear Benchmark.
    Modified from: https://github.com/hearbenchmark/hear-eval-kit/blob/855964977238e89dfc76394aa11c37010edb6f20/heareval/predictions/task_predictions.py#L142

    Args:
        input_size (int): input_size
        output_size (int): output_size
        hidden_size (int): hidden size across all layers. Default: 1024
        hidden_layers (int): number of hidden layers, all in :code:`hidden_size`. Default: 2
        norm_after_activation (bool): whether to norm after activation. Default: False
        dropout (float): dropout ratio. Default: 0.1
        initialization (str): initialization method name available in :obj:`torch.nn.init`
        hidden_norm (str): normalization method name available in :obj:`torch.nn`
        pooling_type (str): the pooling class name in :obj:`s3prl.nn.pooling`. Default: MeanPooling
        pooling_conf (dict): the arguments for initializing the pooling class.
            Default: empty dict
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 1024,
        hidden_layers: int = 2,
        norm_after_activation: bool = False,
        dropout: float = 0.1,
        initialization: str = "xavier_uniform_",
        hidden_norm: str = "BatchNorm1d",
        pooling_type: str = None,
        pooling_conf: dict = None,
    ):
        super().__init__()
        self._input_size = input_size
        self._output_size = output_size
        initialization = getattr(torch.nn.init, initialization)
        hidden_norm = getattr(torch.nn, hidden_norm)

        curdim = input_size
        if pooling_type is not None:
            pooling_cls = getattr(pooling, pooling_type)
            self.pooling = pooling_cls(input_size, **(pooling_conf or {}))
            curdim = self.pooling.output_size

        hidden_modules: List[torch.nn.Module] = []
        last_activation = "linear"
        if hidden_layers:
            for i in range(hidden_layers):
                linear = torch.nn.Linear(curdim, hidden_size)
                initialization(
                    linear.weight,
                    gain=torch.nn.init.calculate_gain(last_activation),
                )
                hidden_modules.append(linear)
                if not norm_after_activation:
                    hidden_modules.append(hidden_norm(hidden_size))
                hidden_modules.append(torch.nn.Dropout(dropout))
                hidden_modules.append(torch.nn.ReLU())
                if norm_after_activation:
                    hidden_modules.append(hidden_norm(hidden_size))
                curdim = hidden_size
                last_activation = "relu"

            self.hidden = torch.nn.Sequential(*hidden_modules)
        else:
            self.hidden = torch.nn.Identity()  # type: ignore
        self.projection = torch.nn.Linear(curdim, output_size)

        initialization(
            self.projection.weight, gain=torch.nn.init.calculate_gain(last_activation)
        )

    @property
    def input_size(self) -> int:
        return self._input_size

    @property
    def output_size(self) -> int:
        return self._output_size

    def forward(self, x, x_len) -> torch.Tensor:
        """
        Args:
            x (torch.FloatTensor): (batch_size, seq_len, input_size)
            x_len (torch.LongTensor): (batch_size, )

        Returns:
            tuple:

            1. y (torch.FloatTensor)
            2. y_len (torch.LongTensor)

            if :code:`pooling_type` is None, :code:`y` is (batch_size, seq_len, output_size) and :code:`y_len` is (batch_size, )
            if not None, :code:`y` is (batch_size, output_size) and :code:`y_len` is (batch_size, ) in all 1s.
        """
        if hasattr(self, "pooling"):
            x = self.pooling(x, x_len)
            x_len = x.new_ones(len(x))

        shape = x.shape
        if len(shape) == 3:
            bs, ts, hidden_size = x.shape
            x = x.reshape(bs * ts, hidden_size)

        x = self.hidden(x)
        x = self.projection(x)

        if len(shape) == 3:
            x = x.reshape(bs, ts, -1)

        return x, x_len
