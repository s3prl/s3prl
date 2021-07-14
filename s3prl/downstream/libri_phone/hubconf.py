import os
import torch

from s3prl.utility.download import _urls_to_filepaths
from .upstream_expert import UpstreamExpert as _UpstreamExpert


def libri_posteriorgram_local(ckpt, *args, **kwargs):
    """
        The model from local ckpt
            ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, *args, **kwargs)


def libri_posteriorgram_url(ckpt, refresh=False, *args, **kwargs):
    """
        The model from URL
            ckpt (str): URL
    """
    return libri_posteriorgram_local(_urls_to_filepaths(ckpt, refresh=refresh), *args, **kwargs)


def libri_posteriorgram(refresh=False, *args, **kwargs):
    """
        The default model
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs['ckpt'] = 'http://140.112.21.12:8000/libri_posteriorgram/dev-acc-0.7979.ckpt'
    return libri_posteriorgram_url(refresh=refresh, *args, **kwargs)
