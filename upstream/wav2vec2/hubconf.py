import os
import torch

from utility.download import _urls_to_filepaths
from .expert import UpstreamExpert as _UpstreamExpert


def wav2vec2_local(ckpt, *args, **kwargs):
    """
        The model from local ckpt
            ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, *args, **kwargs)


def wav2vec2_url(ckpt, refresh=False, *args, **kwargs):
    """
        The model from google drive id
            ckpt (str): URL
            refresh (bool): whether to download ckpt/config again if existed
    """
    return wav2vec2_local(_urls_to_filepaths(ckpt, refresh=refresh), *args, **kwargs)


def wav2vec2(refresh=False, *args, **kwargs):
    """
        The default model - Base
            refresh (bool): whether to download ckpt/config again if existed
    """
    return wav2vec2_base(refresh=refresh, *args, **kwargs)


def wav2vec2_base(refresh=False, *args, **kwargs):
    """
        The Base model
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs['ckpt'] = 'https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt'
    return wav2vec2_url(refresh=refresh, *args, **kwargs)


def wav2vec2_large(refresh=False, *args, **kwargs):
    """
        The Large model
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs['ckpt'] = 'https://dl.fbaipublicfiles.com/fairseq/wav2vec/libri960_big.pt'
    return wav2vec2_url(refresh=refresh, *args, **kwargs)
