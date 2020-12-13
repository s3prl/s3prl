import os
import torch

from hubconf import _gdown
from .expert import UpstreamExpert as _UpstreamExpert


def tera(ckpt, *args, **kwargs):
    """
    The TERA model
        ckpt (str): kwargs, path to the pretrained weights of the model.
    """
    assert os.path.isfile(ckpt)
    upstream = _UpstreamExpert(ckpt)
    return upstream


def tera_default(refresh=False, *args, **kwargs):
    """
    The TERA model from https://drive.google.com/u/1/uc?id=1A9Fs2k3aekY4_6I2GD4tBtjx_v0mV_k4
        refresh (bool): whether to download ckpt/config again if existed
    """
    url = 'https://drive.google.com/u/1/uc?id=1A9Fs2k3aekY4_6I2GD4tBtjx_v0mV_k4'
    ckpt = _gdown('tera_default.ckpt', url, refresh)
    return tera(ckpt)
