from typing import List, Tuple

import torch
import torch.nn as nn

__all__ = [
    "AbsUpstream",
    "AbsFeaturizer",
    "AbsFrameModel",
    "AbsUtteranceModel",
]


class AbsUpstream(nn.Module):
    @property
    def num_layer(self) -> int:
        raise NotImplementedError

    @property
    def hidden_sizes(self) -> List[int]:
        raise NotImplementedError

    @property
    def downsample_rates(self) -> List[int]:
        raise NotImplementedError

    def forward(
        self, wavs: torch.FloatTensor, wavs_len: torch.LongTensor
    ) -> Tuple[List[torch.FloatTensor], List[torch.LongTensor]]:
        raise NotImplementedError


class AbsFeaturizer(nn.Module):
    @property
    def output_size(self) -> int:
        raise NotImplementedError

    @property
    def downsample_rate(self) -> int:
        raise NotImplementedError

    def forward(
        self, all_hs: List[torch.FloatTensor], all_hs_len: List[torch.LongTensor]
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        raise NotImplementedError


class AbsFrameModel(nn.Module):
    @property
    def input_size(self) -> int:
        raise NotImplementedError

    @property
    def output_size(self) -> int:
        raise NotImplementedError

    def forward(
        self, x: torch.FloatTensor, x_len: torch.LongTensor
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        raise NotImplementedError


class AbsUtteranceModel(nn.Module):
    @property
    def input_size(self) -> int:
        raise NotImplementedError

    @property
    def output_size(self) -> int:
        raise NotImplementedError

    def forward(
        self, x: torch.FloatTensor, x_len: torch.LongTensor
    ) -> torch.FloatTensor:
        raise NotImplementedError
