from multiprocessing.sharedctypes import Value
from typing import List

import torch

from s3prl.nn.pooling import MeanPooling, TemporalStatisticsPooling


class HearFullyConnectedPrediction(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_dim: int = 1024,
        hidden_layers: int = None,
        norm_after_activation: bool = False,
        dropout: float = 0.1,
        initialization=torch.nn.init.xavier_uniform_,
        hidden_norm=torch.nn.BatchNorm1d,
        pooling: str = None,
        **kwds,
    ):
        super().__init__()
        self._input_size = input_size
        self._output_size = output_size

        if pooling == "mean":
            self.pooling = MeanPooling()
        elif pooling == "statistics":
            self.pooling = TemporalStatisticsPooling()
        elif isinstance(pooling, str):
            raise ValueError(f"Unsupported pooling type {pooling}")

        hidden_modules: List[torch.nn.Module] = []
        curdim = input_size
        # Honestly, we don't really know what activation preceded
        # us for the final embedding.
        last_activation = "linear"
        if hidden_layers:
            for i in range(hidden_layers):
                linear = torch.nn.Linear(curdim, hidden_dim)
                initialization(
                    linear.weight,
                    gain=torch.nn.init.calculate_gain(last_activation),
                )
                hidden_modules.append(linear)
                if not norm_after_activation:
                    hidden_modules.append(hidden_norm(hidden_dim))
                hidden_modules.append(torch.nn.Dropout(dropout))
                hidden_modules.append(torch.nn.ReLU())
                if norm_after_activation:
                    hidden_modules.append(hidden_norm(hidden_dim))
                curdim = hidden_dim
                last_activation = "relu"

            self.hidden = torch.nn.Sequential(*hidden_modules)
        else:
            self.hidden = torch.nn.Identity()  # type: ignore
        self.projection = torch.nn.Linear(curdim, output_size)

        initialization(
            self.projection.weight, gain=torch.nn.init.calculate_gain(last_activation)
        )

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._output_size

    def forward(self, x: torch.Tensor, x_len) -> torch.Tensor:
        shape = x.shape
        if len(shape) == 3:
            bs, ts, hidden_size = x.shape
            x = x.reshape(bs * ts, hidden_size)

        x = self.hidden(x)
        x = self.projection(x)

        if len(shape) == 3:
            x = x.reshape(bs, ts, -1)
            if hasattr(self, "pooling"):
                x = self.pooling(x, x_len)
                x_len = x.new_ones(len(x))

        return x, x_len
