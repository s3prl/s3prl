# Copyright (c) Facebook, Inc. All Rights Reserved

# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/multires_hubert/hubconf.py ]
#   Synopsis     [ the Multiresolution HuBERT torch hubconf ]
#   Author       [ S3PRL / Jiatong Shi ]
"""*********************************************************************************************"""

# isort: off
import logging
import os
import time
from pathlib import Path

from filelock import FileLock
from s3prl.util.download import _urls_to_filepaths

from .convert import load_and_convert_fairseq_ckpt
from .expert import UpstreamExpert as _UpstreamExpert

# isort: on

logger = logging.getLogger(__name__)

NEW_ENOUGH_SECS = 2.0


def multires_hubert_custom(
    ckpt: str,
    refresh: bool = False,
    **kwargs,
):

    if ckpt.startswith("http"):
        ckpt = _urls_to_filepaths(ckpt, refresh=refresh)
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt=ckpt, **kwargs)


def multires_hubert_local(*args, **kwargs):
    return multires_hubert_custom(*args, **kwargs)


def multires_hubert_base(refresh=False, **kwargs):
    """
    The monolingual base model
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs[
        "ckpt"
    ] = "https://huggingface.co/s3prl/mr_hubert/resolve/main/mrhubert_mono_base.pt"
    return multires_hubert_custom(refresh=refresh, **kwargs)


def multires_hubert_large(refresh=False, **kwargs):
    """
    The monolingual base model
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs[
        "ckpt"
    ] = "https://huggingface.co/s3prl/mr_hubert/resolve/main/mrhubert_mono_large.pt"
    return multires_hubert_custom(refresh=refresh, **kwargs)


def multires_hubert_multilingual_base(refresh=False, **kwargs):
    """
    The multilingual base model
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs["ckpt"] = "https://huggingface.co/s3prl/mr_hubert/resolve/main/multi_base.pt"
    return multires_hubert_custom(refresh=refresh, **kwargs)


def multires_hubert_multilingual_large400k(refresh=False, **kwargs):
    """
    The multilingual large model (400k steps)
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs[
        "ckpt"
    ] = "https://huggingface.co/s3prl/mr_hubert/resolve/main/multi_large_400k.pt"
    return multires_hubert_custom(refresh=refresh, **kwargs)


def multires_hubert_multilingual_large600k(refresh=False, **kwargs):
    """
    The multilingual large model (600k steps)
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs[
        "ckpt"
    ] = "https://huggingface.co/s3prl/mr_hubert/resolve/main/multi_large_600k.pt"
    return multires_hubert_custom(refresh=refresh, **kwargs)
