# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/wavlm/hubconf.py ]
#   Synopsis     [ the WavLM torch hubconf ]
#   Author       [ Microsoft ]
"""*********************************************************************************************"""


import os

from s3prl.util.download import _urls_to_filepaths

from .expert import UpstreamExpert as _UpstreamExpert


def wavlm_local(ckpt, *args, **kwargs):
    """
    The model from local ckpt
        ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, *args, **kwargs)


def wavlm_url(ckpt, refresh=False, *args, **kwargs):
    """
    The model from google drive id
        ckpt (str): URL
        refresh (bool): whether to download ckpt/config again if existed
    """
    return wavlm_local(_urls_to_filepaths(ckpt, refresh=refresh), *args, **kwargs)


def wavlm(refresh=False, *args, **kwargs):
    """
    The default model - Base-Plus
        refresh (bool): whether to download ckpt/config again if existed
    """
    return wavlm_base_plus(refresh=refresh, *args, **kwargs)


def wavlm_base(refresh=False, *args, **kwargs):
    """
    The Base model
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs[
        "ckpt"
    ] = "https://huggingface.co/s3prl/converted_ckpts/resolve/main/wavlm_base.pt"

    return wavlm_url(refresh=refresh, *args, **kwargs)


def wavlm_base_plus(refresh=False, *args, **kwargs):
    """
    The Base-Plus model
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs[
        "ckpt"
    ] = "https://huggingface.co/s3prl/converted_ckpts/resolve/main/wavlm_base_plus.pt"

    return wavlm_url(refresh=refresh, *args, **kwargs)


def wavlm_large(refresh=False, *args, **kwargs):
    """
    The Large model
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs[
        "ckpt"
    ] = "https://huggingface.co/s3prl/converted_ckpts/resolve/main/wavlm_large.pt"

    return wavlm_url(refresh=refresh, *args, **kwargs)
