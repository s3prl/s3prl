import os

import torch

from s3prl.util.download import _urls_to_filepaths

from .expert import UpstreamExpert as _UpstreamExpert


def data2vec_custom(ckpt: str, refresh: bool = False, **kwargs):
    if ckpt.startswith("http"):
        ckpt = _urls_to_filepaths(ckpt, refresh=refresh)

    return _UpstreamExpert(ckpt, **kwargs)


def data2vec_local(*args, **kwargs):
    return data2vec_custom(*args, **kwargs)


def data2vec_url(*args, **kwargs):
    return data2vec_custom(*args, **kwargs)


def data2vec(refresh=False, *args, **kwargs):
    """
    The default model - Base
        refresh (bool): whether to download ckpt/config again if existed
    """
    return data2vec_base_960(refresh=refresh, *args, **kwargs)


def data2vec_base_960(refresh=False, *args, **kwargs):
    """
    The Base model
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs[
        "ckpt"
    ] = "https://huggingface.co/s3prl/converted_ckpts/resolve/main/audio_base_ls.pt"
    return data2vec_custom(refresh=refresh, *args, **kwargs)


def data2vec_large_ll60k(refresh=False, *args, **kwargs):
    """
    The Large model trained on Libri-light 60k hours of data
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs[
        "ckpt"
    ] = "https://huggingface.co/s3prl/converted_ckpts/resolve/main/vox_pretrained.pt"
    return data2vec_custom(refresh=refresh, *args, **kwargs)
