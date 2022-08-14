import os
import torch

from s3prl.util.download import _urls_to_filepaths
from .upstream_expert import UpstreamExpert as _UpstreamExpert


def timit_posteriorgram_local(ckpt, *args, **kwargs):
    """
        The model from local ckpt
            ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, *args, **kwargs)


def timit_posteriorgram_url(ckpt, refresh=False, *args, **kwargs):
    """
        The model from URL
            ckpt (str): URL
    """
    return timit_posteriorgram_local(_urls_to_filepaths(ckpt, refresh=refresh), *args, **kwargs)


def timit_posteriorgram(refresh=False, *args, **kwargs):
    """
        The default model
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs['ckpt'] = 'https://www.dropbox.com/s/fb2hkvetp26wges/convbank.ckpt?dl=1'
    return timit_posteriorgram_url(refresh=refresh, *args, **kwargs)
