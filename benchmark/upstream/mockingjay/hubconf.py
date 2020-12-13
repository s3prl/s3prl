import os
import torch

from hubconf import _gdriveids_to_filepaths
from .expert import UpstreamExpert as _UpstreamExpert

MODEL = 'Mockingjay'


def mockingjay(ckpt, *args, **kwargs):
    f"""
        The {MODEL} model from local ckpt
            ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    upstream = _UpstreamExpert(ckpt)
    return upstream


def mockingjay_gdriveid(ckpt, refresh=False, *args, **kwargs):
    f"""
        The {MODEL} model from google drive id
            ckpt (str): The unique id in the google drive share link
            refresh (bool): whether to download ckpt/config again if existed
    """
    return mockingjay(_gdriveids_to_filepaths(ckpt, refresh=refresh))


def mockingjay_default(refresh=False, *args, **kwargs):
    f"""
        The default {MODEL} model
            refresh (bool): whether to download ckpt/config again if existed
    """
    return mockingjay_gdriveid('1MoF_poVUaL3tKe1tbrQuDIbsC38IMpnH', refresh=refresh)
