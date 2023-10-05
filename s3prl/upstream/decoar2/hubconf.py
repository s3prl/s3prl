import os

import torch

from s3prl.util.download import _urls_to_filepaths

from .expert import UpstreamExpert as _UpstreamExpert


def decoar2_custom(ckpt: str, refresh=False, *args, **kwargs):
    if ckpt.startswith("http"):
        ckpt = _urls_to_filepaths(ckpt, refresh=refresh)

    return _UpstreamExpert(ckpt, *args, **kwargs)


def decoar2_local(*args, **kwargs):
    """
    The model from local ckpt
        ckpt (str): PATH
        feature_selection (str): 'c' (default) or 'z'
    """
    return decoar2_custom(*args, **kwargs)


def decoar2_url(*args, **kwargs):
    """
    The model from URL
        ckpt (str): URL
    """
    return decoar2_custom(*args, **kwargs)


def decoar2(*args, refresh=False, **kwargs):
    """
    The apc standard model on 360hr
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs[
        "ckpt"
    ] = "https://huggingface.co/s3prl/converted_ckpts/resolve/main/checkpoint_decoar2.pt"
    return decoar2_url(*args, refresh=refresh, **kwargs)
