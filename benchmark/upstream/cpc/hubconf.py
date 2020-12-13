import os
import torch

from utility.download import _urls_to_filepaths
from .expert import UpstreamExpert as _UpstreamExpert


def cpc(ckpt, *args, **kwargs):
    """
        The model from local ckpt
            ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, *args, **kwargs)


def cpc_url(ckpt, refresh=False, *args, **kwargs):
    """
        The model from URL
            ckpt (str): URL
    """
    return cpc(_urls_to_filepaths(ckpt), *args, **kwargs)


def cpc_default(refresh=False, *args, **kwargs):
    """
        The model from official repository
    """
    kwargs['ckpt'] = 'https://dl.fbaipublicfiles.com/librilight/CPC_checkpoints/60k_epoch4-d0f474de.pt'
    return cpc_url(refresh=refresh, *args, **kwargs)
