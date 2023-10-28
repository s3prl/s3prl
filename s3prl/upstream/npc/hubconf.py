# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/npc/hubconf.py ]
#   Synopsis     [ the npc torch hubconf ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


import os

from s3prl.util.download import _urls_to_filepaths

from .expert import UpstreamExpert as _UpstreamExpert


def npc_local(ckpt, *args, **kwargs):
    """
    The model from local ckpt
        ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, *args, **kwargs)


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
    return npc_360hr(refresh=refresh, *args, **kwargs)


def npc_360hr(refresh=False, *args, **kwargs):
    """
    The npc standard model on 360hr
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs[
        "ckpt"
    ] = "https://huggingface.co/leo19941227/apc_series/resolve/main/npc_360hr.ckpt"
    return npc_url(refresh=refresh, *args, **kwargs)


def npc_960hr(refresh=False, *args, **kwargs):
    """
    The npc standard model on 960hr
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs[
        "ckpt"
    ] = "https://huggingface.co/leo19941227/apc_series/resolve/main/npc_960hr.ckpt"
    return npc_url(refresh=refresh, *args, **kwargs)
