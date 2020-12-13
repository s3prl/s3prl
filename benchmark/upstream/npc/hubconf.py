import os
import torch

from hubconf import _gdown
from .expert import UpstreamExpert as _UpstreamExpert


def npc(ckpt, config, *args, **kwargs):
    """
    The Mockingjay model
        ckpt (str): kwargs, path to the pretrained weights of the model.
    """
    assert os.path.isfile(ckpt)
    assert os.path.isfile(config)
    upstream = _UpstreamExpert(ckpt, config)
    return upstream


def npc_default(refresh=False, *args, **kwargs):
    """
    The default npc model
        refresh (bool): whether to download ckpt/config again if existed
    """
    ckpt_url = 'https://drive.google.com/uc?id=1oNmdVEcFMtNt4Rrllxs3ulrEk5TJdioW'
    config_url = 'https://drive.google.com/uc?id=1ZrfpE0K6us_daQeJTwVENGZn8a4MiPaB'
    ckpt = _gdown('npc_default.ckpt', ckpt_url, refresh)
    config = _gdown('npc_default.yaml', config_url, refresh)
    return npc(ckpt, config)
