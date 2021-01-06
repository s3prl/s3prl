import os
import torch

from utility.download import _gdriveids_to_filepaths
from upstream.mockingjay.expert import UpstreamExpert as _UpstreamExpert


def mockingjay(ckpt, *args, **kwargs):
    """
        The model from local ckpt
            ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, *args, **kwargs)


def mockingjay_gdriveid(ckpt, refresh=False, *args, **kwargs):
    """
        The model from google drive id
            ckpt (str): The unique id in the google drive share link
            refresh (bool): whether to download ckpt/config again if existed
    """
    return mockingjay(_gdriveids_to_filepaths(ckpt, refresh=refresh), *args, **kwargs)


def mockingjay_default(refresh=False, *args, **kwargs):
    """
        The default model
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs['ckpt'] = '1MoF_poVUaL3tKe1tbrQuDIbsC38IMpnH'
    return mockingjay_gdriveid(refresh=refresh, *args, **kwargs)
