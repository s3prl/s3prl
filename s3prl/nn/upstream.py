from multiprocessing.sharedctypes import Value
from typing import Union

import torch

from s3prl import Output, hub

from . import NNModule

CHECK_ITERATION = 10


class S3PRLUpstream(NNModule):
    def __init__(
        self,
        name: str,
        feature_selection: str = "hidden_states",
        layer_selection: int = None,
        layer_drop: Union[str, float] = 0.0,
    ):
        """
        Args:
            name (str):
                can be "apc", "hubert", "wav2vec2", the names supported by s3prl.hub
            feature_selection (str):
                cat be "hidden_states" for the defaults, or "ASR", "SID", "ER" ... etc
            layer_selection (int):
                a specific layer selected after feature_selection
            layer_drop (float):
                By default, set upstream's layer_drop (if exists) to 0 to prevent inconsistant
                number of layers during multiple inference.
                if layer_drop == "original", then use the initial layer_drop value as released
        """
        super().__init__()
        self.upstream = getattr(hub, name)()
        self.feature_selection = feature_selection
        self.layer_selection = layer_selection

        if hasattr(self.upstream, "layer_drop"):
            if layer_drop == "original":
                self.upstream.set_layer_drop()
            elif isinstance(layer_drop, float):
                self.upstream.set_layer_drop(layer_drop)
            else:
                raise ValueError("Unsupported layer_drop value")

        output_sizes = []
        num_hidden_states = []
        for _ in range(CHECK_ITERATION):
            output_size, num_hidden_state = self._get_hidden_states_statistics(
                self.upstream, self.feature_selection
            )
            output_sizes.append(output_size)
            num_hidden_states.append(num_hidden_state)

        assert len(set(output_sizes)) == 1
        assert (
            len(set(num_hidden_states)) == 1
        ), f"multiple inference get different number of layers: {num_hidden_states}"

        self._output_size = output_sizes[0]
        self._num_hidden_state = num_hidden_states[0]

    @staticmethod
    def _get_hidden_states_statistics(upstream, feature_selection):
        pseudo_hidden_states = upstream([torch.randn(16000)])[feature_selection]
        output_size = pseudo_hidden_states[0].size(-1)
        num_hidden_state = len(pseudo_hidden_states)
        return output_size, num_hidden_state

    @property
    def num_hidden_state(self):
        return self._num_hidden_state

    @property
    def input_size(self):
        return 1

    @property
    def downsample_rate(self) -> int:
        return self.upstream.get_downsample_rates(self.feature_selection)

    @property
    def output_size(self):
        return self._output_size

    def forward(self, x, x_len, **kwargs):
        """
        Args:
            x (torch.Tensor): (B, T, 1)
            x_len (torch.LongTensor): (B, )

        Return:
            hidden_states (list): a list of torch.Tensor (B, T, H)
            hidden_states_len (list): a list of torch.LongTensor (B, )
        """
        assert x.dim() == 3
        assert x.size(-1) == self.input_size

        xs = [w.view(-1)[:l] for w, l in zip(x, x_len)]
        hidden_states = self.upstream(xs)[self.feature_selection]
        downsample_rate = self.upstream.get_downsample_rates(self.feature_selection)

        if isinstance(hidden_states, torch.Tensor):
            hidden_states = [hidden_states]

        hidden_states_len = [
            torch.LongTensor(
                [
                    min(int(l.item() / downsample_rate), hidden_states[0].size(1))
                    for l in x_len
                ]
            ).to(x.device)
        ] * len(hidden_states)

        if self.layer_selection is not None:
            hidden_states = hidden_states[self.layer_selection]
            hidden_states_len = hidden_states_len[self.layer_selection]

        return Output(
            hidden_states=hidden_states,
            hidden_states_len=hidden_states_len,
        )
