"""
S3PRL Upstream Collection and some utilities

Authors:
  * Leo 2022
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from s3prl import hub
from s3prl.util.pseudo_data import get_pseudo_wavs

__all__ = [
    "S3PRLUpstream",
    "Featurizer",
    "UpstreamDownstreamModel",
]

MIN_SECOND = 0.05
SAMPLE_RATE = 16000


class S3PRLUpstream(nn.Module):
    """
    This is an easy interface for using all the models in S3PRL.

    Args:
        name (str):
            can be "apc", "hubert", "wav2vec2". See :obj:`available_names` for all the supported names

        path_or_url (str):
            The source of the checkpoint. Might be a local path or a URL

        refresh (bool): (default, False)
            If false, only downlaod checkpoint if not yet downloaded before.
            If true, force to re-download the checkpoint.

        extra_conf (dict):
            The extra arguments for each specific upstream, the available options are
            shown in each upstream section

    .. note::

        When using **S3PRLUpstream** with :code:`refresh=True` and multiprocessing (e.g. DDP),
        the checkpoint will only be downloaded once, and the other processes will simply
        re-use the newly downloaded checkpoint, instead of re-downloading on every processes,
        which can be very time/bandwidth consuming.

    Example::

        >>> import torch
        >>> from s3prl.nn import S3PRLUpstream
        ...
        >>> model = S3PRLUpstream("hubert")
        >>> model.eval()
        ...
        >>> with torch.no_grad():
        ...     wavs = torch.randn(2, 16000 * 2)
        ...     wavs_len = torch.LongTensor([16000 * 1, 16000 * 2])
        ...     all_hs, all_hs_len = model(wavs, wavs_len)
        ...
        >>> for hs, hs_len in zip(all_hs, all_hs_len):
        ...     assert isinstance(hs, torch.FloatTensor)
        ...     assert isinstance(hs_len, torch.LongTensor)
        ...
        ...     batch_size, max_seq_len, hidden_size = hs.shape
        ...     assert hs_len.dim() == 1
    """

    @classmethod
    def available_names(cls, only_registered_ckpt: bool = False) -> List[str]:
        """
        All the available names supported by this S3PRLUpstream

        Args:
            only_registered_ckpt (bool):
                ignore entry names which require to give `path_or_url`.
                That is, the entry names without the registered checkpoint sources.
                These names end with :code:`_local` (for local path), :code:`_url`
                (for URL) or :code:`_custom` (auto-determine path or URL)
        """
        return hub.options(only_registered_ckpt)

    def __init__(
        self,
        name: str,
        path_or_url: str = None,
        refresh: bool = False,
        normalize: bool = False,
        extra_conf: dict = None,
    ):
        super().__init__()
        self.upstream = getattr(hub, name)(
            ckpt=path_or_url, refresh=refresh, **(extra_conf or {})
        )
        self.normalize = normalize

        self.upstream.eval()
        with torch.no_grad():
            hs = self.upstream(get_pseudo_wavs())["hidden_states"]
        self.upstream.train()
        self._num_layers = len(hs)

        self._hidden_sizes = []
        for h in hs:
            self._hidden_sizes.append(h.size(-1))

        downsample_rates = self.upstream.get_downsample_rates("hidden_states")
        if isinstance(downsample_rates, int):
            self._downsample_rates = [downsample_rates] * self._num_layers
        elif isinstance(downsample_rates, (tuple, list)):
            self._downsample_rates = downsample_rates
        else:
            raise ValueError

    @property
    def num_layers(self) -> int:
        """
        Number of hidden sizes. All the upstream have a deterministic
        number of layers. That is, layer drop is turned off by default.
        """
        return self._num_layers

    @property
    def downsample_rates(self) -> List[int]:
        """
        Downsampling rate from 16000 Hz audio of each layer.
        Usually, all layers have the same downsampling rate,
        but might not be the case for some advanced upstreams.
        """
        return self._downsample_rates

    @property
    def hidden_sizes(self) -> List[int]:
        """
        The hidden size of each layer
        """
        return self._hidden_sizes

    def _match_length(self, xs, target_max_len: int):
        xs_max_len = xs.size(1)

        if xs_max_len > target_max_len:
            assert xs_max_len // target_max_len == 1, f"{xs_max_len}, {target_max_len}"
            xs = xs[:, :target_max_len, :]

        elif xs_max_len < target_max_len:
            assert target_max_len // xs_max_len == 1, f"{target_max_len}, {xs_max_len}"
            xs = torch.cat(
                (xs, xs[:, -1:, :].repeat(1, target_max_len - xs_max_len, 1)), dim=1
            )

        return xs

    def forward(self, wavs: torch.FloatTensor, wavs_len: torch.LongTensor):
        """
        Args:
            wavs (torch.FloatTensor): (batch_size, seqlen) or (batch_size, seqlen, 1)
            wavs_len (torch.LongTensor): (batch_size, )

        Return:
            List[torch.FloatTensor], List[torch.LongTensor]

            1. all the layers of hidden states: List[ (batch_size, max_seq_len, hidden_size) ]
            2. the valid length for each hidden states: List[ (batch_size, ) ]
        """
        if wavs.dim() == 3:
            wavs = wavs.squeeze(-1)

        original_wavs_len = wavs_len
        if max(original_wavs_len) < MIN_SECOND * SAMPLE_RATE:
            padded_samples = int(MIN_SECOND * SAMPLE_RATE) - max(original_wavs_len)
            wavs = torch.cat(
                (wavs, wavs.new_zeros(wavs.size(0), padded_samples)),
                dim=1,
            )
            wavs_len = wavs_len + padded_samples

        wavs_list = []
        for wav, wav_len in zip(wavs, wavs_len):
            wavs_list.append(wav[:wav_len])

        hidden_states = self.upstream(wavs_list)["hidden_states"]
        assert isinstance(hidden_states, (list, tuple))
        assert (
            len(hidden_states) == self.num_layers
        ), f"{len(hidden_states)}, {self.num_layers}"

        max_wav_len = int(max(wavs_len))
        all_hs = []
        all_lens = []
        for h, stride in zip(hidden_states, self.downsample_rates):
            expected_max_h_len = max_wav_len // stride + 1
            h = self._match_length(h, expected_max_h_len)
            assert h.size(1) == expected_max_h_len

            h_len = torch.div(original_wavs_len, stride, rounding_mode="floor") + 1
            h = h[:, : max(h_len), :]
            if self.normalize:
                h = F.layer_norm(h, h.shape[-1:])

            all_hs.append(h)
            all_lens.append(h_len)

        return all_hs, all_lens


class Featurizer(nn.Module):
    """
    Featurizer take the :obj:`S3PRLUpstream`'s multiple layer of hidden_states and
    reduce (standardize) them into a single hidden_states, to connect with downstream NNs.

    This basic Featurizer expects all the layers to have same stride and hidden_size
    When the input upstream only have a single layer of hidden states, use that directly.
    If multiple layers are presented, add a trainable weighted-sum on top of those layers.

    Args:
        upstream (:obj:`S3PRLUpstream`):
            the upstream to extract features, this upstream is used only for initialization
            and will not be kept in this Featurizer object
        layer_selections (List[int]):
            To select a subset of hidden states from the given upstream by layer ids (0-index)
            If None (default), than all the layer of hidden states are selected
        normalize (bool):
            Whether to apply layer norm on all the hidden states before weighted-sum
            This can help convergence in some cases, but not used in SUPERB to ensure the
            fidelity of each upstream's extracted representation.

    Example::

        >>> import torch
        >>> from s3prl.nn import S3PRLUpstream
        ...
        >>> model = S3PRLUpstream("hubert")
        >>> model.eval()
        ...
        >>> with torch.no_grad():
        ...     wavs = torch.randn(2, 16000 * 2)
        ...     wavs_len = torch.LongTensor([16000 * 1, 16000 * 2])
        ...     all_hs, all_hs_len = model(wavs, wavs_len)
        ...
        >>> featurizer = Featurizer(model)
        >>> hs, hs_len = featurizer(all_hs, all_hs_len)
        ...
        >>> assert isinstance(hs, torch.FloatTensor)
        >>> assert isinstance(hs_len, torch.LongTensor)
        >>> batch_size, max_seq_len, hidden_size = hs.shape
        >>> assert hs_len.dim() == 1
    """

    def __init__(
        self,
        upstream: S3PRLUpstream,
        layer_selections: List[int] = None,
        normalize: bool = False,
    ):
        super().__init__()
        assert len(set(upstream.hidden_sizes)) == 1
        assert len(set(upstream.downsample_rates)) == 1
        self._output_size = upstream.hidden_sizes[0]
        self._downsample_rate = upstream.downsample_rates[0]
        self.normalize = normalize

        if upstream.num_layers > 1:
            if layer_selections is not None:
                assert upstream.num_layers >= len(layer_selections)
                self.layer_selections = sorted(layer_selections)
            else:
                self.layer_selections = list(range(upstream.num_layers))
            self.weights = nn.Parameter(torch.zeros(len(self.layer_selections)))

    @property
    def output_size(self) -> int:
        """
        The hidden size of the final weighted-sum output
        """
        return self._output_size

    @property
    def downsample_rate(self) -> int:
        """
        The downsample rate (from 16k Hz waveform) of the final weighted-sum output
        """
        return self._downsample_rate

    def _weighted_sum(self, all_hs, all_lens):
        assert len(all_hs) == len(all_lens) > 1
        for l in all_lens[1:]:
            torch.allclose(all_lens[0], l)
        stacked_hs = torch.stack(all_hs, dim=0)

        if self.normalize:
            stacked_hs = F.layer_norm(stacked_hs, (stacked_hs.shape[-1],))

        _, *origin_shape = stacked_hs.shape
        stacked_hs = stacked_hs.view(len(self.layer_selections), -1)
        norm_weights = F.softmax(self.weights, dim=-1)
        weighted_hs = (norm_weights.unsqueeze(-1) * stacked_hs).sum(dim=0)
        weighted_hs = weighted_hs.view(*origin_shape)

        return weighted_hs, all_lens[0]

    def forward(
        self, all_hs: List[torch.FloatTensor], all_lens: List[torch.LongTensor]
    ):
        """
        Args:
            all_hs (List[torch.FloatTensor]): List[ (batch_size, seq_len, hidden_size) ]
            all_lens (List[torch.LongTensor]): List[ (batch_size, ) ]

        Return:
            torch.FloatTensor, torch.LongTensor

            1. The weighted-sum result, (batch_size, seq_len, hidden_size)
            2. the valid length of the result, (batch_size, )
        """
        if len(all_hs) == 1:
            return all_hs[0], all_lens[0]

        all_hs = [h for idx, h in enumerate(all_hs) if idx in self.layer_selections]
        all_lens = [l for idx, l in enumerate(all_lens) if idx in self.layer_selections]
        hs, hs_len = self._weighted_sum(all_hs, all_lens)
        return hs, hs_len


class UpstreamDownstreamModel(nn.Module):
    def __init__(
        self,
        upstream: S3PRLUpstream,
        featurizer: Featurizer,
        downstream,
        upstream_trainable: bool = False,
    ):
        super().__init__()
        self.upstream = upstream
        self.featurizer = featurizer
        self.downstream = downstream
        self.upstream_trainable = upstream_trainable

    @property
    def input_size(self):
        return 1

    @property
    def downsample_rate(self):
        return self.featurizer.downsample_rate

    @property
    def output_size(self):
        return self.downstream.output_size

    def forward(self, wav, wav_len, *args, **kwargs):
        with torch.set_grad_enabled(self.upstream_trainable):
            if not self.upstream_trainable:
                self.upstream.eval()
            hs, hs_len = self.upstream(wav, wav_len)

        h, h_len = self.featurizer(hs, hs_len)
        return self.downstream(h, h_len, *args, **kwargs)
