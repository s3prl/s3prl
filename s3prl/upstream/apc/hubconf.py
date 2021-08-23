# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/apc/hubconf.py ]
#   Synopsis     [ the apc torch hubconf ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


import os

from s3prl.utility.download import _urls_to_filepaths
from .expert import UpstreamExpert as _UpstreamExpert


def apc_local(ckpt, *args, **kwargs):
    """
        The model from local ckpt
            ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, *args, **kwargs)


def apc_url(ckpt, refresh=False, *args, **kwargs):
    """
        The model from URL
            ckpt (str): URL
    """
    return apc_local(_urls_to_filepaths(ckpt, refresh=refresh), *args, **kwargs)


def apc(refresh=False, *args, **kwargs):
    """
        The default model
            refresh (bool): whether to download ckpt/config again if existed
    """
    return apc_360hr(refresh=refresh, *args, **kwargs)


def apc_360hr(refresh=False, *args, **kwargs):
    """
        The apc standard model on 360hr
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs['ckpt'] = 'https://www.dropbox.com/s/mcq82c0x62h9004/apc_default.ckpt?dl=0'
    return apc_url(refresh=refresh, *args, **kwargs)


def apc_960hr(refresh=False, *args, **kwargs):
    """
        The apc standard model on 960hr
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs['ckpt'] = 'https://www.dropbox.com/s/mmfx3opdr4lz25n/apc_960hr.ckpt?dl=0'
    return apc_url(refresh=refresh, *args, **kwargs)