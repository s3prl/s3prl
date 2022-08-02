import random
from typing import Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from s3prl import Output, Container, hub

from . import NNModule

CHECK_ITERATION = 10
SAMPLE_RATE = 16000


def get_pseudo_list_wavs():
    lengths = [random.randint(SAMPLE_RATE * 1, SAMPLE_RATE * 3) for _ in range(4)]
    wavs = [torch.randn(l) for l in lengths]
    return wavs


def get_pseudo_wavs_and_lengths():
    wavs = get_pseudo_list_wavs()
    wavs_len = [len(w) for w in wavs]
    wavs = pad_sequence(wavs, batch_first=True)
    return wavs.unsqueeze(-1), torch.LongTensor(wavs_len)


class S3PRLUpstream(NNModule):
    """
    This is an easy interface for using all the S3PRL SSL models.

    Args:
        name (str):
            can be "apc", "hubert", "wav2vec2". See above model information for all the supported names
        feature_selection (str):
            **"hidden_states"** is the default which extracts all layers, different :code:`name`
            comes with different supported :code:`feature_selection` options. See the above
            model information for the supported options. (Only a few have options other than **hidden_states**)
        layer_drop (float):
            By default, set upstream's layer_drop (if exists) to 0 to prevent inconsistant
            number of layers during multiple inference.
            if layer_drop is None, then use the initial layer_drop value as released
        refresh (bool): (default, False)
            If true, force to re-download the checkpoint
    """

    def __init__(
        self,
        name: str,
        ckpt: str = None,
        feature_selection: str = "hidden_states",
        layer_drop: Union[str, float] = 0.0,
        refresh: bool = False,
        legacy: bool = False,
    ):
        super().__init__()
        self.upstream = getattr(hub, name)(ckpt=ckpt, refresh=refresh, legacy=legacy)
        self.feature_selection = feature_selection
        self._downsample_rate = self.upstream.get_downsample_rates(
            self.feature_selection
        )

        # Some classic models like wav2vec 2.0 has layer-drop
        # Usually we want to turn this off during the inference
        layer_dropped = hasattr(self.upstream, "layer_drop") and hasattr(
            self.upstream, "set_layer_drop"
        )
        if layer_dropped:
            if layer_drop is None:
                self.upstream.set_layer_drop()
            elif isinstance(layer_drop, float):
                self.upstream.set_layer_drop(layer_drop)
            else:
                raise ValueError("Unsupported layer_drop value")

        if not (layer_dropped and layer_drop > 0):
            assert _deterministic_num_layers(self._get_hidden_state_sizes)

    def _get_hidden_state_sizes(self):
        pseudo_hidden_states = self.upstream(get_pseudo_list_wavs())[
            self.feature_selection
        ]
        output_sizes = [h.size(-1) for h in pseudo_hidden_states]
        return output_sizes

    @property
    def downsample_rate(self):
        return self._downsample_rate

    def forward(self, x, x_len, **kwds):
        """
        Different layers of hidden states might have different hidden sizes
        Also, if layer drop is on, might output different number of layers.
        Use self.eval or set :obj:`layer_drop` to 0.0 to ensure deterministic
        number of layers.

        Args:
            x (torch.Tensor): (B, T, 1)
            x_len (torch.LongTensor): (B, )

        Return:
            :obj:`s3prl.base.container.Container`

            hidden_states (List[torch.FloatTensor]):
                [(B, T / :obj:`downsample_rate`, hidden_size1), ...]
            hidden_states_len (torch.LongTensor): (B, )
        """
        assert x.dim() == 3
        assert x.size(-1) == 1

        xs = [w.view(-1)[:l] for w, l in zip(x, x_len)]
        hidden_states = self.upstream(xs)[self.feature_selection]

        if isinstance(hidden_states, torch.Tensor):
            hidden_states = [hidden_states]

        hidden_states_len = torch.LongTensor(
            [
                min(round(l.item() / self.downsample_rate), hidden_states[0].size(1))
                for l in x_len
            ]
        ).to(x.device)

        return Output(
            hidden_states=hidden_states,
            hidden_states_len=hidden_states_len,
        )


class WeightedSum(nn.Module):
    def __init__(self, num_layer: int) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(num_layer))

    def forward(self, x: torch.Tensor, x_len: torch.LongTensor = None):
        assert (
            x.dim() == 4
        ), "x should have dimensions: (num_layers, batch_size, seq_len, hidden_size)"
        weights = F.softmax(self.weights, dim=-1)
        x = (x * weights.view(-1, 1, 1, 1)).sum(dim=0)
        return Output(output=x, output_len=x_len)


class UpstreamDriver(NNModule):
    """
    Args:
        mode (str): (default, 'weighted_sum') weighted_sum, finetune_default
    """

    def __init__(
        self,
        cls: type = S3PRLUpstream,
        cfg: dict = None,
        freeze_upstream: bool = True,
        normalize: bool = False,
        weighted_sum: bool = True,
        layer_selections: List[int] = None,
    ):
        super().__init__()
        cfg = cfg or dict()
        self.upstream = cls(**cfg)
        self.freeze_upstream = freeze_upstream
        self.normalize = normalize
        self.weighted_sum = weighted_sum
        self.layer_selections = layer_selections

        if freeze_upstream:
            self.upstream.requires_grad_(False)

        if weighted_sum or layer_selections is not None:
            assert _deterministic_num_layers(self._get_hidden_state_sizes)

        if weighted_sum:
            if layer_selections is None:
                self.weights = WeightedSum(len(self._get_hidden_state_sizes()))
            else:
                self.weights = WeightedSum(len(layer_selections))

        output_sizes = []
        for _ in range(CHECK_ITERATION):
            y, y_len = self(*get_pseudo_wavs_and_lengths()).slice(2)
            output_sizes.append(y.size(-1))
        assert len(set(output_sizes)) == 1
        self._output_size = output_sizes[0]

    @property
    def input_size(self):
        return 1

    @property
    def output_size(self):
        return self._output_size

    @property
    def downsample_rate(self):
        return self.upstream.downsample_rate

    def _get_hidden_state_sizes(self):
        pseudo_hs, h_len = self.upstream(*get_pseudo_wavs_and_lengths()).slice(2)
        output_sizes = [h.size(-1) for h in pseudo_hs]
        return output_sizes

    def forward(self, x, x_len, **unused):
        """
        Args:
            x (torch.FloatTensor): (batch_size, seq_len, 1)
            x_len (torch.LongTensor): (batch_size)
        """
        if self.freeze_upstream:
            self.upstream.eval()

        hs, h_len = self.upstream(x, x_len).slice(2)

        if self.normalize:
            hs = [F.layer_norm(h, (h.size(-1),)) for h in hs]

        if self.weighted_sum:
            if self.layer_selections is not None:
                hs = [hs[layer_id] for layer_id in self.layer_selections]
            hs = torch.stack(hs, dim=0)
            y = self.weights(hs).slice(1)
        else:
            y = hs[-1]
        return Output(output=y, output_len=h_len)


class S3PRLUpstreamDriver(UpstreamDriver):
    def __init__(
        self,
        name: str,
        ckpt: str = None,
        feature_selection: str = "hidden_states",
        layer_drop: Union[str, float] = 0.0,
        refresh: bool = False,
        freeze_upstream: bool = True,
        normalize: bool = False,
        weighted_sum: bool = True,
        layer_selections: List[int] = None,
        legacy: bool = True,  # FIXME (Leo): This is a temporary solution
    ):
        super().__init__(
            S3PRLUpstream,
            dict(
                name=name,
                ckpt=ckpt,
                feature_selection=feature_selection,
                layer_drop=layer_drop,
                refresh=refresh,
                legacy=legacy,
            ),
            freeze_upstream,
            normalize,
            weighted_sum,
            layer_selections,
        )


def _deterministic_num_layers(get_layer_output_sizes: callable):
    num_hidden_states = []
    for _ in range(CHECK_ITERATION):
        output_sizes = get_layer_output_sizes()
        num_hidden_states.append(len(output_sizes))

    return len(set(num_hidden_states)) == 1


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
        upstream: UpstreamDriver,
        downstream: DownstreamExample,
    ):
        """
        Args:
            upstream (S3PRLUpstreamDriver)
            downstream (DownstreamExample)
        """
        super().__init__()
        self.upstream = (
            upstream if isinstance(upstream, nn.Module) else Container(upstream)()
        )
        self.downstream = (
            downstream
            if isinstance(downstream, nn.Module)
            else Container(downstream)(input_size=self.upstream.output_size)
        )
        assert upstream.input_size == 1
        assert upstream.output_size == downstream.input_size

    @property
    def input_size(self):
        return self.upstream.input_size

    @property
    def output_size(self):
        return self.downstream.output_size

    @property
    def feat_frame_shift(self):
        return self.upstream.downsample_rate

    def forward(self, wav, wav_len, *args, **kwargs):
        h, h_len = self.upstream(wav, wav_len).slice(2)
        return self.downstream(h, h_len, *args, **kwargs)
