import os
import torch

from hubconf import _gdown
from .expert import UpstreamExpert as _UpstreamExpert


def mockingjay(ckpt, *args, **kwargs):
    """
    The Mockingjay model
        ckpt (str): kwargs, path to the pretrained weights of the model.
    """
    assert os.path.isfile(ckpt)
    upstream = _UpstreamExpert(ckpt)
    return upstream


def mockingjay_default(use_cache=True, *args, **kwargs):
    """
    The Mockingjay model from https://drive.google.com/u/1/uc?id=1MoF_poVUaL3tKe1tbrQuDIbsC38IMpnH
        use_cache (bool): whether to download ckpt/config again if existed
    """
    url = 'https://drive.google.com/u/1/uc?id=1MoF_poVUaL3tKe1tbrQuDIbsC38IMpnH'
    ckpt = _gdown('mockingjay_default.ckpt', url, use_cache)
    return mockingjay(ckpt)
