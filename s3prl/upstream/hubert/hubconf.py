# Copyright (c) Facebook, Inc. All Rights Reserved

# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/hubert/hubconf.py ]
#   Synopsis     [ the HuBERT torch hubconf ]
#   Author       [ Kushal Lakhotia ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os

# -------------#
from s3prl.utility.download import _urls_to_filepaths
from .expert import UpstreamExpert as _UpstreamExpert


def hubert_local(ckpt, *args, **kwargs):
    """
    The model from local ckpt
        ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, *args, **kwargs)


def hubert_url(ckpt, refresh=False, *args, **kwargs):
    """
    The model from google drive id
        ckpt (str): URL
        refresh (bool): whether to download ckpt/config again if existed
    """
    return hubert_local(
        _urls_to_filepaths(ckpt, refresh=refresh), *args, **kwargs
    )


def hubert(refresh=False, *args, **kwargs):
    """
    The default model - Base
        refresh (bool): whether to download ckpt/config again if existed
    """
    return hubert_base(refresh=refresh, *args, **kwargs)


def hubert_base(refresh=False, *args, **kwargs):
    """
    The Base model
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs["ckpt"] = "https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt"
    return hubert_url(refresh=refresh, *args, **kwargs)


def hubert_large_ll60k(refresh=False, *args, **kwargs):
    """
    The Large model
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs["ckpt"] = "https://dl.fbaipublicfiles.com/hubert/hubert_large_ll60k.pt"
    return hubert_url(refresh=refresh, *args, **kwargs)


def hubert_base_robust_mgr(refresh=False, *args, **kwargs):
    """
    The Base model, continually trained with Libri 960 hr with Musan noise, Gaussian noise and Reverberation.
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs["ckpt"] = "https://huggingface.co/kphuang68/HuBERT_base_robust_mgr/resolve/main/HuBERT_base_robust_mgr_best_loss_2.7821.pt"
    return hubert_url(refresh=refresh, *args, **kwargs)