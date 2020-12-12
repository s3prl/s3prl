import torch

from hubconf import _gdown
from .expert import UpstreamExpert as _UpstreamExpert


def tera(ckpt=None, *args, **kwargs):
    """
    The TERA model
        ckpt (str): kwargs, path to the pretrained weights of the model.
    """
    upstream = _UpstreamExpert(ckpt)
    return upstream


def tera_default(use_cache=True, *args, **kwargs):
    """
    The TERA model from https://drive.google.com/u/1/uc?id=1A9Fs2k3aekY4_6I2GD4tBtjx_v0mV_k4
        use_cache (bool): whether to download ckpt/config again if existed
    """
    url = 'https://drive.google.com/u/1/uc?id=1A9Fs2k3aekY4_6I2GD4tBtjx_v0mV_k4'
    ckpt = _gdown('tera_default.ckpt', url, use_cache)
    return tera(ckpt)
