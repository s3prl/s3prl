import os
import torch

from utility.download import _gdriveids_to_filepaths
from .expert import UpstreamExpert as _UpstreamExpert


def mockingjay(ckpt, *args, **kwargs):
    f"""
        The model from local ckpt
            ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    upstream = _UpstreamExpert(ckpt)
    return upstream


def mockingjay_gdriveid(ckpt, refresh=False, *args, **kwargs):
    f"""
        The model from google drive id
            ckpt (str): The unique id in the google drive share link
            refresh (bool): whether to download ckpt/config again if existed
    """
    return mockingjay(_gdriveids_to_filepaths(ckpt, refresh=refresh))


def mockingjay_default(refresh=False, *args, **kwargs):
    f"""
        The default model
            refresh (bool): whether to download ckpt/config again if existed
    """
    return mockingjay_gdriveid('1MoF_poVUaL3tKe1tbrQuDIbsC38IMpnH', refresh=refresh)
