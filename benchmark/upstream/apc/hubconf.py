import os
import torch

from utility.download import _gdriveids_to_filepaths
from .expert import UpstreamExpert as _UpstreamExpert


def apc(ckpt, *args, **kwargs):
    """
        The model from local ckpt
            ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, *args, **kwargs)


def apc_gdriveid(ckpt, refresh=False, *args, **kwargs):
    """
        The model from google drive id
            ckpt (str): The unique id in the google drive share link
            refresh (bool): whether to download ckpt/config again if existed
    """
    return apc(_gdriveids_to_filepaths(ckpt, refresh=refresh), *args, **kwargs)


def apc_default(refresh=False, *args, **kwargs):
    """
        The default model
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs['ckpt'] = '1JXb3cJ38JMOQkorwGN_s2pwK_ig_u90A'
    return apc_gdriveid(refresh=refresh, *args, **kwargs)
