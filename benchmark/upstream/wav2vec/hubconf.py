import os
import torch

from utility.download import _urls_to_filepaths
from .expert import UpstreamExpert as _UpstreamExpert


def wav2vec(ckpt, config, *args, **kwargs):
    f"""
        The model from local ckpt
            ckpt (str): PATH
            config (str): PATH
    """
    assert os.path.isfile(ckpt)
    assert os.path.isfile(config)
    return _UpstreamExpert(ckpt, config)


def wav2vec_url(ckpt, refresh=False, *args, **kwargs):
    f"""
        The model from google drive id
            ckpt (str): URL
            config (str): PATH
            refresh (bool): whether to download ckpt/config again if existed
    """
    return wav2vec(_urls_to_filepaths(ckpt, refresh=refresh), *args, **kwargs)


def wav2vec_default(refresh=False, *args, **kwargs):
    f"""
        The default model - Large model
            config (str): PATH
            refresh (bool): whether to download ckpt/config again if existed
    """
    return wav2vec_large(refresh=refresh, *args, **kwargs)


def wav2vec_large(refresh=False, *args, **kwargs):
    f"""
        The Large model
            config (str): PATH
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs['ckpt'] = 'https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_large.pt'
    return wav2vec_url(refresh=refresh, *args, **kwargs)
