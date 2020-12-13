import os
import torch
from functools import partial as _partial

from hubconf import _url_wrapper, _gdriveid_wrapper
from .expert import UpstreamExpert as _UpstreamExpert

MODEL = 'TERA'


def tera(ckpt, *args, **kwargs):
    f"""
        The {MODEL} model from local ckpt
            ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    upstream = _UpstreamExpert(ckpt)
    return upstream


tera_url = _partial(_url_wrapper, cls=tera)
tera_url.__doc__ =\
f"""
    The {MODEL} model from url
        ckpt (str): URL
        refresh (bool): whether to download ckpt/config again if existed
"""


tera_gdriveid = _partial(_gdriveid_wrapper, cls=tera)
tera_gdriveid.__doc__ =\
f"""
    The {MODEL} model from google drive id
        ckpt (str): The unique id in the google drive share link
        refresh (bool): whether to download ckpt/config again if existed
"""


def tera_default(refresh=False, *args, **kwargs):
    f"""
        The default {MODEL} model
            refresh (bool): whether to download ckpt/config again if existed
    """
    return tera_gdriveid(ckpt='1A9Fs2k3aekY4_6I2GD4tBtjx_v0mV_k4', refresh=refresh)
