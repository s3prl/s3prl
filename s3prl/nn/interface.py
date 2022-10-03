"""
Model interfaces

Authors:
  * Leo 2022
"""

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
    """
    The upstream model should follow this interface. Please subclass it.
    """

    @property
    def num_layer(self) -> int:
        """
        number of hidden states
        """
        raise NotImplementedError

    @property
    def hidden_sizes(self) -> List[int]:
        """
        hidden size of each hidden state
        """
        raise NotImplementedError

    @property
    def downsample_rates(self) -> List[int]:
        """
        downsample rate from 16 KHz waveforms for each hidden state
        """
        raise NotImplementedError

    def forward(
        self, wavs: torch.FloatTensor, wavs_len: torch.LongTensor
    ) -> Tuple[List[torch.FloatTensor], List[torch.LongTensor]]:
        """
        Args:
            wavs (torch.FloatTensor): (batch_size, seq_len, 1)
            wavs_len (torch.LongTensor): (batch_size, )

        Returns:
            tuple:

            1. all_hs (List[torch.FloatTensor]): all the hidden states
            2. all_hs_len (List[torch.LongTensor]): the lengths for all the hidden states
        """
        raise NotImplementedError


class AbsFeaturizer(nn.Module):
    """
    The featurizer should follow this interface. Please subclass it.
    The featurizer's mission is to reduce (standardize) the multiple hidden
    states from :obj:`AbsUpstream` into a single hidden state, so that
    the downstream model can use it as a conventional representation.
    """

    @property
    def output_size(self) -> int:
        """
        The output size after hidden states reduction
        """
        raise NotImplementedError

    @property
    def downsample_rate(self) -> int:
        """
        The downsample rate from 16 KHz waveform of the reduced single hidden state
        """
        raise NotImplementedError

    def forward(
        self, all_hs: List[torch.FloatTensor], all_hs_len: List[torch.LongTensor]
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """
        Args:
            all_hs (List[torch.FloatTensor]): all the hidden states
            all_hs_len (List[torch.LongTensor]): the lengths for all the hidden states

        Returns:
            tuple:

            1. hs (torch.FloatTensor)
            2. hs_len (torch.LongTensor)
        """
        raise NotImplementedError


class AbsFrameModel(nn.Module):
    """
    The frame-level model interface.
    """

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
    """
    The utterance-level model interface, which pools the temporal dimension.
    """

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
