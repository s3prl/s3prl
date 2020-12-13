import os
import torch

from hubconf import _gdown
from .expert import UpstreamExpert as _UpstreamExpert


def apc(ckpt, config, *args, **kwargs):
    """
    The Mockingjay model
        ckpt (str): kwargs, path to the pretrained weights of the model.
    """
    assert os.path.isfile(ckpt)
    assert os.path.isfile(config)
    upstream = _UpstreamExpert(ckpt, config)
    return upstream


def apc_default(refresh=False, *args, **kwargs):
    """
    The default apc model
        refresh (bool): whether to download ckpt/config again if existed
    """
    ckpt_url = 'https://drive.google.com/uc?id=17EUmcitnDCZ1vBTDR7Qq_JP3j-z1cO74'
    config_url = 'https://drive.google.com/uc?id=1N7oAecCBAEqqSS4QPXs_gN2RkSfd5i5-'
    ckpt = _gdown('apc_default.ckpt', ckpt_url, refresh)
    config = _gdown('apc_default.yaml', config_url, refresh)
    return apc(ckpt, config)
