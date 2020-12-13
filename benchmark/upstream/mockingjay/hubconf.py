import os
import torch
from functools import partial, update_wrapper

from hubconf import _url_wrapper, _gdriveid_wrapper
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


mockingjay_url = partial(_url_wrapper, cls=mockingjay)
mockingjay_url.__doc__ =\
f"""
    The {MODEL} model from url
        ckpt (str): URL
        refresh (bool): whether to download ckpt/config again if existed
"""


mockingjay_gdriveid = partial(_gdriveid_wrapper, cls=mockingjay)
mockingjay_gdriveid.__doc__ =\
f"""
    The {MODEL} model from google drive id
        ckpt (str): The unique id in the google drive share link
        refresh (bool): whether to download ckpt/config again if existed
"""


def mockingjay_default(refresh=False, *args, **kwargs):
    f"""
        The default {MODEL} model
            refresh (bool): whether to download ckpt/config again if existed
    """
    return mockingjay_gdriveid(ckpt='1MoF_poVUaL3tKe1tbrQuDIbsC38IMpnH', refresh=refresh)
