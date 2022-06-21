import torch
import torch.nn as nn
import torch.nn.functional as F

from s3prl import Output

from . import NNModule
from .upstream import S3PRLUpstream


class DownstreamExample(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_size: int = 768
        self.output_size: int = 4

    def forward(self, xs, xs_len, **kwargs) -> Output:
        return Output(output=torch.randn(xs.size(0), self.output_size))


class UpstreamDownstreamModel(NNModule):
    def __init__(
        self,
        upstream: S3PRLUpstream,
        downstream: DownstreamExample,
        weighted_sum: bool = True,
        upstream_trainable: bool = False,
        layer_norm: bool = False,
    ):
        """
        Args:
            upstream (S3PRLUpstream)
            downstream (DownstreamExample)
            weighted_sum (bool):
                if True, add trainable weights on top of the extracted upstream representation
            upstream_trainable (bool):
                if False, prevent gradient from propogating through upstream model, and set the
                    upstream model in eval mode.
                if True, let the gradient propogate through upstream model as how it was defined
                    outside this module.
            layer_norm (bool):
                whether to do layer_norm on top of the extracted upstream hidden_states
        """
        super().__init__()
        self.upstream = upstream
        self.weighted_sum = weighted_sum
        self.upstream_trainable = upstream_trainable
        self.layer_norm = layer_norm

        if not self.upstream_trainable:
            self.upstream.requires_grad_(False)

        if self.weighted_sum:
            self.weights = nn.Parameter(torch.zeros(self.upstream.num_hidden_state))

        self.downstream = downstream
        assert upstream.input_size == 1
        assert upstream.output_size == downstream.input_size

    @property
    def input_size(self):
        return self.upstream.input_size

    @property
    def output_size(self):
        return self.downstream.output_size

    def forward(self, wav, wav_len, *args, **kwargs):
        hidden_states, hidden_states_len = self.upstream(wav, wav_len).slice(2)

        if self.layer_norm:
            for index in range(len(hidden_states)):
                hidden_size = hidden_states[index].size(-1)
                hidden_states[index] = F.layer_norm(
                    hidden_states[index], (hidden_size,)
                )

        if self.weighted_sum and len(hidden_states) > 1:
            weights = F.softmax(self.weights, dim=-1)
            assert len(set([h.size(-1) for h in hidden_states])) == 1
            stacked_hidden_states = torch.stack(hidden_states, dim=0)
            hidden_state = (stacked_hidden_states * weights.view(-1, 1, 1, 1)).sum(
                dim=0
            )
        else:
            hidden_state = hidden_states[-1]

        assert (
            hidden_state.size(-1)
            == self.upstream.output_size
            == self.downstream.input_size
        )
        return self.downstream(hidden_state, hidden_states_len, *args, **kwargs)
