# Copyright (c) Facebook, Inc. All Rights Reserved

# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/hubert/hubconf.py ]
#   Synopsis     [ the HuBERT torch hubconf ]
#   Author       [ Kushal Lakhotia ]
"""*********************************************************************************************"""


import os

from s3prl.util.download import _urls_to_filepaths

from .expert import LegacyUpstreamExpert as _LegacyUpstreamExpert
from .expert import UpstreamExpert as _UpstreamExpert


def hubert_custom(
    ckpt: str, *args, legacy: bool = False, refresh: bool = False, **kwargs
):
    if ckpt.startswith("http"):
        ckpt = _urls_to_filepaths(ckpt, refresh=refresh)

    assert os.path.isfile(ckpt)
    if legacy:
        return _LegacyUpstreamExpert(ckpt, *args, **kwargs)
    else:
        return _UpstreamExpert(ckpt, *args, **kwargs)


def hubert_local(*args, **kwargs):
    return hubert_custom(*args, **kwargs)


def hubert_url(*args, **kwargs):
    return hubert_custom(*args, **kwargs)


def hubert(refresh=False, *args, **kwargs):
    """
    The default model - Base
        refresh (bool): whether to download ckpt/config again if existed
    """
    return hubert_base(refresh=refresh, *args, **kwargs)


def hubert_base(refresh=False, legacy=False, **kwargs):
    """
    The Base model
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs["ckpt"] = "https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt"
    if not legacy:
        kwargs[
            "ckpt"
        ] = "https://huggingface.co/s3prl/converted_ckpts/resolve/main/hubert_base_ls960.pt"
    return hubert_custom(refresh=refresh, legacy=legacy, **kwargs)


def hubert_large_ll60k(refresh=False, legacy=False, **kwargs):
    """
    The Large model
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs["ckpt"] = "https://dl.fbaipublicfiles.com/hubert/hubert_large_ll60k.pt"
    if not legacy:
        kwargs[
            "ckpt"
        ] = "https://huggingface.co/s3prl/converted_ckpts/resolve/main/hubert_large_ll60k.pt"
    return hubert_custom(refresh=refresh, legacy=legacy, **kwargs)


def hubert_base_robust_mgr(refresh=False, legacy=False, **kwargs):
    """
    The Base model, continually trained with Libri 960 hr with Musan noise, Gaussian noise and Reverberation.
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs[
        "ckpt"
    ] = "https://huggingface.co/kphuang68/HuBERT_base_robust_mgr/resolve/main/HuBERT_base_robust_mgr_best_loss_2.7821.pt"
    if not legacy:
        kwargs[
            "ckpt"
        ] = "https://huggingface.co/s3prl/converted_ckpts/resolve/main/HuBERT_base_robust_mgr_best_loss_2.7821.pt"
    return hubert_custom(refresh=refresh, legacy=legacy, **kwargs)
