import os
import torch

from hubconf import _url_preprocessor, _gdriveid_preprocessor
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


def mockingjay_url(ckpt, refresh=False, *args, **kwargs):
    f"""
        The {MODEL} model from url
            ckpt (str): URL
            refresh (bool): whether to download ckpt/config again if existed
    """
    return mockingjay(_url_preprocessor(ckpt, refresh=refresh))


def mockingjay_gdriveid(ckpt, refresh=False, *args, **kwargs):
    f"""
        The {MODEL} model from google drive id
            ckpt (str): The unique id in the google drive share link
            refresh (bool): whether to download ckpt/config again if existed
    """
    return mockingjay(_gdriveid_preprocessor(ckpt, refresh=refresh))


def mockingjay_default(refresh=False, *args, **kwargs):
    f"""
        The default {MODEL} model
            refresh (bool): whether to download ckpt/config again if existed
    """
    return mockingjay_gdriveid('1MoF_poVUaL3tKe1tbrQuDIbsC38IMpnH', refresh=refresh)
