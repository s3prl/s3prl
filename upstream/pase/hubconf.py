import os
import torch

from utility.download import _gdriveids_to_filepaths, _urls_to_filepaths
from .expert import UpstreamExpert as _UpstreamExpert


def pase_local(ckpt, model_config, **kwargs):
    """
        The model from local ckpt
            ckpt (str): PATH
            model_config (str): PATH
    """
    assert os.path.isfile(ckpt)
    assert os.path.isfile(model_config)
    return _UpstreamExpert(ckpt, model_config, **kwargs)


def pase_url(ckpt, model_config, refresh=False, **kwargs):
    """
        The model from URL
            ckpt (str): URL
            model_config (str): URL
    """
    ckpt = _urls_to_filepaths(ckpt, refresh=refresh)
    model_config = _urls_to_filepaths(model_config, refresh=refresh)
    return pase_local(ckpt, model_config, **kwargs)


def pase_plus(refresh=False, **kwargs):
    """
        The default model
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs['ckpt'] = 'http://140.112.21.12:8000/pase/FE_e199.ckpt'
    kwargs['model_config'] = 'http://140.112.21.12:8000/pase/PASE%2B.cfg'
    return pase_url(refresh=refresh, **kwargs)
