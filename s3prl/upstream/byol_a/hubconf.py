# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/byol_a/hubconf.py ]
#   Synopsis     [ the BYOL-A torch hubconf ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""

import os

from s3prl.util.download import _urls_to_filepaths

from .expert import UpstreamExpert as _UpstreamExpert

# FIXME (Leo): These are wrong according to the original author


def byol_a_local(ckpt, model_config=None, *args, **kwargs):
    """
    The model from local ckpt
        ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    if model_config is not None:
        assert os.path.isfile(model_config)
    return _UpstreamExpert(ckpt, model_config, *args, **kwargs)


def byol_a_url(ckpt, refresh=False, *args, **kwargs):
    """
    The model from URL
        ckpt (str): URL
    """
    return byol_a_local(_urls_to_filepaths(ckpt, refresh=refresh), *args, **kwargs)


def byol_a(refresh=False, *args, **kwargs):
    """
    The default model
        refresh (bool): whether to download ckpt/config again if existed
    """
    return byol_a_2048(refresh=refresh, *args, **kwargs)


def byol_a_2048(refresh=False, *args, **kwargs):
    """
    refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs[
        "ckpt"
    ] = "https://github.com/nttcslab/byol-a/raw/master/pretrained_weights/AudioNTT2020-BYOLA-64x96d2048.pth"
    return byol_a_url(refresh=refresh, *args, **kwargs)


def byol_a_1024(refresh=False, *args, **kwargs):
    """
    refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs[
        "ckpt"
    ] = "https://github.com/nttcslab/byol-a/raw/master/pretrained_weights/AudioNTT2020-BYOLA-64x96d1024.pth"
    return byol_a_url(refresh=refresh, *args, **kwargs)


def byol_a_512(refresh=False, *args, **kwargs):
    """
    refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs[
        "ckpt"
    ] = "https://github.com/nttcslab/byol-a/raw/master/pretrained_weights/AudioNTT2020-BYOLA-64x96d512.pth"
    return byol_a_url(refresh=refresh, *args, **kwargs)
