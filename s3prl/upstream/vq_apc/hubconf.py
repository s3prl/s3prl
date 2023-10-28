# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/vq_apc/hubconf.py ]
#   Synopsis     [ the vq apc torch hubconf ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


from ..apc.hubconf import apc_url as vq_apc_url


def vq_apc(refresh=False, *args, **kwargs):
    """
    The default model
        refresh (bool): whether to download ckpt/config again if existed
    """
    return vq_apc_360hr(refresh=refresh, *args, **kwargs)


def vq_apc_360hr(refresh=False, *args, **kwargs):
    """
    The vq-apc standard model on 360hr
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs[
        "ckpt"
    ] = "https://huggingface.co/leo19941227/apc_series/resolve/main/vq_apc_360hr.ckpt"
    return vq_apc_url(refresh=refresh, *args, **kwargs)


def vq_apc_960hr(refresh=False, *args, **kwargs):
    """
    The vq-apc standard model on 960hr
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs[
        "ckpt"
    ] = "https://huggingface.co/leo19941227/apc_series/resolve/main/vq_apc_960hr.ckpt"
    return vq_apc_url(refresh=refresh, *args, **kwargs)
