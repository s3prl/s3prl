# Copyright (c) Facebook, Inc. All Rights Reserved

# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/wav2vec/expert.py ]
#   Synopsis     [ the wav2vec wrapper ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


import argparse
from packaging import version

import torch
from torch.nn.utils.rnn import pad_sequence

import fairseq
from fairseq.models.wav2vec import Wav2VecModel

from ..interfaces import UpstreamBase

SAMPLE_RATE = 16000
EXAMPLE_SEC = 5


class UpstreamExpert(UpstreamBase):
    """
    The wav2vec wrapper
    """

    def __init__(self, ckpt, **kwargs):
        super().__init__(**kwargs)
        if version.parse(fairseq.__version__) > version.parse("0.10.2"):
            cp = torch.load(ckpt)
            args = cp["args"]
            base_wav2vec_architecture(args)
            self.model = Wav2VecModel.build_model(args, task=None)
            self.model.load_state_dict(cp["model"])
        elif version.parse(fairseq.__version__) == version.parse("0.10.2"):
            cp = torch.load(ckpt)
            self.model = Wav2VecModel.build_model(cp["args"], task=None)
            self.model.load_state_dict(cp["model"])
        else:
            raise NotImplementedError

        if len(self.hooks) == 0:
            self.add_hook(
                "self.model.feature_extractor",
                lambda input, output: output.transpose(1, 2),
            )
            self.add_hook(
                "self.model.feature_aggregator",
                lambda input, output: output.transpose(1, 2),
            )
            module_name = "self.model.feature_aggregator.conv_layers"
            for conv_id in range(len(eval(module_name)) - 1):
                self.add_hook(
                    f"{module_name}[{conv_id + 1}]",
                    lambda input, output: input[0].transpose(1, 2),
                )

    def get_downsample_rates(self, key: str) -> int:
        return 160

    def forward(self, wavs):
        """
        Code snippet modified from fairseq
        """
        result = {}

        padded_wav = pad_sequence(wavs, batch_first=True)
        features = self.model.feature_extractor(padded_wav)
        result["z"] = features.transpose(1, 2).contiguous()

        if self.model.vector_quantizer:
            q_res = self.model.vector_quantizer(features, produce_targets=True)
            result["codewords"] = q_res["x"].transpose(1, 2).contiguous()
            result["codeids"] = q_res["targets"]
            features = q_res["x"]

        x = self.model.dropout_feats(features)
        x = self.model.feature_aggregator(x)

        result["c"] = x.transpose(1, 2).contiguous()
        result["default"] = result["c"]

        # The keys "hidden_states" and "last_hidden_state" are handled by UpstreamBase's hooks
        return result


def base_wav2vec_architecture(args):
    conv_feature_layers = "[(512, 10, 5)]"
    conv_feature_layers += " + [(512, 8, 4)]"
    conv_feature_layers += " + [(512, 4, 2)] * 3"
    args.conv_feature_layers = getattr(args, "conv_feature_layers", conv_feature_layers)

    args.conv_aggregator_layers = getattr(
        args, "conv_aggregator_layers", "[(512, 3, 1)] * 9"
    )

    args.prediction_steps = getattr(args, "prediction_steps", 12)
    args.num_negatives = getattr(args, "num_negatives", 1)
    args.sample_distance = getattr(args, "sample_distance", None)
    args.cross_sample_negatives = getattr(args, "cross_sample_negatives", 0)

    args.dropout = getattr(args, "dropout", 0.0)
    args.dropout_features = getattr(args, "dropout_features", 0.0)
    args.dropout_agg = getattr(args, "dropout_agg", 0.0)
    args.encoder = getattr(args, "encoder", "cnn")
    args.aggregator = getattr(args, "aggregator", "cnn")

    args.skip_connections_feat = getattr(args, "skip_connections_feat", False)
    args.skip_connections_agg = getattr(args, "skip_connections_agg", False)
    args.residual_scale = getattr(args, "residual_scale", 0.5)

    args.gru_dim = getattr(args, "gru_dim", 512)

    args.no_conv_bias = getattr(args, "no_conv_bias", False)
    args.agg_zero_pad = getattr(args, "agg_zero_pad", False)

    args.log_compression = getattr(args, "log_compression", False)

    args.balanced_classes = getattr(args, "balanced_classes", False)
    args.infonce = getattr(args, "infonce", False)
    args.project_features = getattr(args, "project_features", "none")

    args.non_affine_group_norm = getattr(args, "non_affine_group_norm", False)

    args.offset = getattr(args, "offset", "auto")

    args.activation = getattr(args, "activation", "relu")

    args.vq_type = getattr(args, "vq_type", "none")
    args.vq_vars = getattr(args, "vq_vars", 320)
    args.vq_groups = getattr(args, "vq_groups", 2)
    args.vq_dim = getattr(args, "vq_dim", 0)
    args.vq_depth = getattr(args, "vq_depth", 1)
    args.combine_groups = getattr(args, "combine_groups", False)
    args.vq_temp = getattr(args, "vq_temp", "(2.0, 0.5, 0.999995)")
    args.vq_gamma = getattr(args, "vq_gamma", 0.25)
