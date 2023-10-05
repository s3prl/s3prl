# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/unispeech_sat/hubconf.py ]
#   Synopsis     [ the UniSpeech-SAT torch hubconf ]
#   Author       [ Microsoft ]
"""*********************************************************************************************"""


import os

from s3prl.util.download import _urls_to_filepaths

from .expert import UpstreamExpert as _UpstreamExpert


def unispeech_sat_local(ckpt, *args, **kwargs):
    """
    The model from local ckpt
        ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, *args, **kwargs)


def unispeech_sat_url(ckpt, refresh=False, *args, **kwargs):
    """
    The model from google drive id
        ckpt (str): URL
        refresh (bool): whether to download ckpt/config again if existed
    """
    return unispeech_sat_local(
        _urls_to_filepaths(ckpt, refresh=refresh), *args, **kwargs
    )


def unispeech_sat(refresh=False, *args, **kwargs):
    """
    The default model - Base-Plus
        refresh (bool): whether to download ckpt/config again if existed
    """
    return unispeech_sat_base_plus(refresh=refresh, *args, **kwargs)


def unispeech_sat_base(refresh=False, *args, **kwargs):
    """
    The Base model
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs[
        "ckpt"
    ] = "https://huggingface.co/s3prl/converted_ckpts/resolve/main/unispeech_sat_base.pt"

    return unispeech_sat_url(refresh=refresh, *args, **kwargs)


def unispeech_sat_base_plus(refresh=False, *args, **kwargs):
    """
    The Base-Plus model
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs[
        "ckpt"
    ] = "https://huggingface.co/s3prl/converted_ckpts/resolve/main/unispeech_sat_base_plus.pt"

    return unispeech_sat_url(refresh=refresh, *args, **kwargs)


def unispeech_sat_large(refresh=False, *args, **kwargs):
    """
    The Large model
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs[
        "ckpt"
    ] = "https://huggingface.co/s3prl/converted_ckpts/resolve/main/unispeech_sat_large.pt"

    return unispeech_sat_url(refresh=refresh, *args, **kwargs)
