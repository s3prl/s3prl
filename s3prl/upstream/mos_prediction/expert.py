from collections import OrderedDict
from typing import List, Union, Dict

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torchaudio.sox_effects import apply_effects_tensor
from itertools import accumulate

import s3prl
from ..interfaces import UpstreamBase, Featurizer
from .model import MosDownstream, MosDownstreamModule
from .utility import unfold_segments


class UpstreamExpert(UpstreamBase):
    def __init__(self, ckpt: str = None, model_config: str = None, **kwargs):
        """
        Args:
            ckpt:
                The checkpoint path for loading your pretrained weights.
                Can be assigned by the -k option in run_downstream.py

            model_config:
                The config path for constructing your model.
                Might not needed if you also save that in your checkpoint file.
                Can be assigned by the -g option in run_downstream.py
        """
        # Pass kwargs into UpstreamBase to enable features shared across upstreams
        super().__init__(**kwargs)

        # The model needs to be a nn.Module for finetuning, not required for representation extraction
        self.checkpoint = torch.load(ckpt, map_location="cpu")

        self.upstream_type = kwargs["upstream"]

        self.mos_upstream = self._get_mos_upstream()
        self.mos_featurizer = self._get_mos_featurizer()
        self.mos_downstream = self._get_mos_downstream()

        self.segments_durations = 1

    def forward(
        self, wavs: List[Tensor]
    ) -> Dict[str, Union[Tensor, List[Tensor], Dict[str, Tensor]]]:
        """
        When the returning Dict contains the List or Dict with more than one Tensor,
        those Tensors should be in the same shape if one wished to weighted sum them.
        """
        wavs_segments = [self.preprocessor(wav) for wav in wavs]

        flattened_wavs_segments = [
            wav_segment
            for wav_segments in wavs_segments
            for wav_segment in wav_segments
        ]
        wav_segments_lengths = [len(wav_segments) for wav_segments in wavs_segments]
        prefix_sums = list(accumulate(wav_segments_lengths, initial=0))

        features = self.mos_upstream(flattened_wavs_segments)
        features = self.mos_featurizer(flattened_wavs_segments, features)
        features = torch.stack(features)
        segments_scores = self.mos_downstream(features)

        scores = []
        for i in range(len(prefix_sums) - 1):
            current_segment_scores = segments_scores[
                prefix_sums[i] : prefix_sums[i + 1]
            ]
            scores.append(current_segment_scores.mean(dim=-1))

        scores = torch.FloatTensor(scores)

        return {"scores": scores}

    def preprocessor(self, wav):

        wav_segments = unfold_segments(wav, self.segments_durations)
        return wav_segments

    def _get_mos_upstream(self):
        mos_upstream = getattr(s3prl.hub, self.upstream_type)()
        if self.upstream_type == "tera":
            self.checkpoint["Upstream"][
                "transformer.extracter._melscale.fb"
            ] = torch.tensor([])
        mos_upstream.load_state_dict(self.checkpoint["Upstream"])
        return mos_upstream

    def _get_mos_featurizer(self):
        return Featurizer(self.mos_upstream, upstream_device="cpu")

    def _get_mos_downstream(self):
        mos_downstream = MosDownstream(
            upstream_dim=self.mos_featurizer.output_dim,
            projector_dim=self.checkpoint["Config"]["downstream_expert"]["modelrc"][
                "projector_dim"
            ],
            clipping=self.checkpoint["Config"]["downstream_expert"]["modelrc"][
                "clipping"
            ],
            attention_pooling=self.checkpoint["Config"]["downstream_expert"]["modelrc"][
                "attention_pooling"
            ],
        )

        mos_downstream.load_state_dict(self.checkpoint["Downstream"])
        return mos_downstream
