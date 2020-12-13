import os
import torch

from hubconf import _gdown
from .expert import UpstreamExpert as _UpstreamExpert


def vq_apc(ckpt, config, *args, **kwargs):
    """
    The Mockingjay model
        ckpt (str): kwargs, path to the pretrained weights of the model.
    """
    assert os.path.isfile(ckpt)
    assert os.path.isfile(config)
    upstream = _UpstreamExpert(ckpt, config)
    return upstream


def vq_apc_default(refresh=False, *args, **kwargs):
    """
    The default vq_apc model
        refresh (bool): whether to download ckpt/config again if existed
    """
    ckpt_url = 'https://drive.google.com/uc?id=1swpF6nCLU2xVWRmwbt0s2w0BkQ8ys2iy'
    config_url = 'https://drive.google.com/uc?id=1aH13mvds_ZERD5pI0NfOTBtFhbKQL_0_'
    ckpt = _gdown('vq_apc_default.ckpt', ckpt_url, refresh)
    config = _gdown('vq_apc_default.yaml', config_url, refresh)
    return vq_apc(ckpt, config)
