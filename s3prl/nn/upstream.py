import logging
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from s3prl import hub
from s3prl.util.pseudo_data import get_pseudo_wavs

logger = logging.getLogger(__name__)

CHECK_ITERATION = 10
TOLERABLE_SEQLEN_DIFF = 3


class S3PRLUpstream(nn.Module):
    """
    This is an easy interface for using all the S3PRL SSL models

    Args:
        name (str):
            can be "apc", "hubert", "wav2vec2". See above model information for all the supported names

        path_or_url (str):
            The source of the checkpoint. Might be a local path or a URL

        refresh (bool): (default, False)
            If true, force to re-download the checkpoint even if it exists
    """

    @classmethod
    def available_names(cls, only_registered_ckpt: bool = False):
        return hub.options(only_registered_ckpt)

    def __init__(self, name: str, path_or_url: str = None, refresh: bool = False):
        super().__init__()
        self.upstream = getattr(hub, name)(ckpt=path_or_url, refresh=refresh)

        hs = self.upstream(get_pseudo_wavs())["hidden_states"]
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
    def num_layers(self):
        return self._num_layers

    @property
    def downsample_rates(self):
        return self._downsample_rates

    @property
    def hidden_sizes(self):
        return self._hidden_sizes

    def _match_length(self, xs, target_max_len: int):
        xs_max_len = xs.size(1)

        assert abs(target_max_len - xs_max_len) < TOLERABLE_SEQLEN_DIFF
        factor = int(round(target_max_len / xs_max_len))
        assert factor == 1

        if xs_max_len > target_max_len:
            xs = xs[:, :target_max_len, :]

        elif xs_max_len < target_max_len:
            xs = torch.cat(
                (xs, xs[:, -1:, :].repeat(1, target_max_len - xs_max_len, 1)), dim=1
            )

        return xs

    def forward(self, wavs: torch.FloatTensor, wavs_len: torch.LongTensor):
        """
        Args:
            wavs: (batch_size, seqlen)
            wavs_len: (batch_size)
        """
        wavs_list = []
        for wav, wav_len in zip(wavs, wavs_len):
            wavs_list.append(wav[:wav_len])

        hidden_states = self.upstream(wavs_list)["hidden_states"]
        assert isinstance(hidden_states, (list, tuple))
        assert len(hidden_states) == self.num_layers

        max_wav_len = int(max(wavs_len))
        all_hs = []
        all_lens = []
        for h, stride in zip(hidden_states, self.downsample_rates):
            expected_max_h_len = round(max_wav_len / stride)
            h = self._match_length(h, expected_max_h_len)
            assert h.size(1) == expected_max_h_len
            all_hs.append(h)

            h_len = (wavs_len.float() / stride).round().long()
            all_lens.append(h_len)

        return all_hs, all_lens


class Featurizer(nn.Module):
    """
    This basic Featurizer expects all the layers to have same stride and hidden_size
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
        self.normalize = normalize

        if upstream.num_layers > 1:
            if layer_selections is not None:
                assert upstream.num_layers >= len(layer_selections)
                self.layer_selections = sorted(layer_selections)
            else:
                self.layer_selections = list(range(upstream.num_layers))
            self.weights = nn.Parameter(torch.zeros(len(self.layer_selections)))

    @property
    def output_size(self):
        return self._output_size

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
        if len(all_hs) == 1:
            return all_hs[0], all_lens[0]

        all_hs = [h for idx, h in enumerate(all_hs) if idx in self.layer_selections]
        all_lens = [l for idx, l in enumerate(all_lens) if idx in self.layer_selections]
        hs, hs_len = self._weighted_sum(all_hs, all_lens)
        return hs, hs_len
