import os
import torch

from utility.download import _gdriveids_to_filepaths
from .expert import UpstreamExpert as _UpstreamExpert


def npc(ckpt, *args, **kwargs):
    f"""
        The model from local ckpt
            ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, *args, **kwargs)


def npc_gdriveid(ckpt, refresh=False, *args, **kwargs):
    f"""
        The model from google drive id
            ckpt (str): The unique id in the google drive share link
            refresh (bool): whether to download ckpt/config again if existed
    """
    return npc(_gdriveids_to_filepaths(ckpt, refresh=refresh), *args, **kwargs)


def npc_default(refresh=False, *args, **kwargs):
    f"""
        The default model
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs['ckpt'] = '1YS81RctXE8aR8GVmA0cFjkb_ruCQXYip'
    return npc_gdriveid(refresh=refresh, *args, **kwargs)
