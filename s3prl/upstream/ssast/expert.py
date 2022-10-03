# -*- coding: utf-8 -*-
# @Time    : 8/25/21 5:25 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : expert.py

import logging

import torch

from s3prl.upstream.interfaces import SAMPLE_RATE

from .audio import FeatureExtractor

logger = logging.getLogger(__name__)


class UpstreamExpert(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        try:
            import timm

            from .ast_models import ASTModel
        except:
            print(
                "SSAST requires 'timm==0.4.5' to work. Please run 'pip install timm==0.4.5'"
            )
            raise

        model_size = kwargs["model_size"]
        pretrain_path = kwargs["pretrain_path"]
        target_length = kwargs["target_length"]
        self.expected_secs = float(target_length / 100)

        model_size, model_type = model_size.split("_")[0], model_size.split("_")[1]
        self.preprocessor = FeatureExtractor(
            target_length=target_length, apply_cmvn=False
        )
        if model_type == "p":
            logger.info("now train a patch models")
            self.model = ASTModel(
                fshape=16,
                tshape=16,
                fstride=10,
                tstride=10,
                input_tdim=target_length,
                input_fdim=128,
                model_size=model_size,
                pretrain_stage=False,
                load_pretrained_mdl_path=pretrain_path,
            )
            self.vertical_num_patches = (128 - 16) // 10 + 1  # 12
        else:
            logger.info("now train a frame models")
            self.model = ASTModel(
                fshape=128,
                tshape=2,
                fstride=128,
                tstride=1,
                input_tdim=target_length,
                input_fdim=128,
                model_size=model_size,
                pretrain_stage=False,
                load_pretrained_mdl_path=pretrain_path,
            )
            self.vertical_num_patches = (128 - 128) // 128 + 1  # 1

    def get_downsample_rates(self, key: str = None) -> int:
        return 160

    def forward(self, wavs):
        wavs_len = [len(wav) for wav in wavs]
        max_wav_len = max(wavs_len)
        seg_n_sample = int(self.expected_secs * SAMPLE_RATE)
        padded_max_wav_len = (max_wav_len // seg_n_sample + 1) * seg_n_sample
        padded_wavs = [
            torch.cat([wav, wav.new_zeros(padded_max_wav_len - len(wav))])
            for wav in wavs
        ]

        all_hs = []
        for start in range(0, max(len(wav) for wav in padded_wavs), seg_n_sample):
            subwavs = [wav[start : start + seg_n_sample] for wav in padded_wavs]
            features = [self.preprocessor(wav.unsqueeze(0)) for wav in subwavs]
            features = torch.stack(features, dim=0)
            hidden_states, _ = self.model(features)
            all_hs.append(hidden_states)

        all_chunked_hs = list(zip(*all_hs))
        all_hs = []
        max_feat_len = max([l // self.get_downsample_rates() for l in wavs_len])
        for chunk_hs in all_chunked_hs:
            all_hs.append(torch.cat(chunk_hs, dim=1)[:, :max_feat_len, :])

        return {"hidden_states": all_hs}
