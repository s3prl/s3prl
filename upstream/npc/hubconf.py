import os
import torch

from utility.download import _gdriveids_to_filepaths, _urls_to_filepaths
from .expert import UpstreamExpert as _UpstreamExpert


def npc_local(ckpt, *args, **kwargs):
    """
        The model from local ckpt
            ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, *args, **kwargs)


def npc_gdriveid(ckpt, refresh=False, *args, **kwargs):
    """
        The model from google drive id
            ckpt (str): The unique id in the google drive share link
            refresh (bool): whether to download ckpt/config again if existed
    """
    return npc_local(_gdriveids_to_filepaths(ckpt, refresh=refresh), *args, **kwargs)


def npc_url(ckpt, refresh=False, *args, **kwargs):
    """
        The model from URL
            ckpt (str): URL
    """
    return npc_local(_urls_to_filepaths(ckpt, refresh=refresh), *args, **kwargs)


def npc(refresh=False, *args, **kwargs):
    """
        The default model
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs['ckpt'] = 'http://140.112.21.12:8000/npc/npc_360hr.ckpt'
    return npc_url(refresh=refresh, *args, **kwargs)
