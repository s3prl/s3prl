# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/mlm/hubconf.py ]
#   Synopsis     [ the mlm torch hubconf ]
#   Author       [ Andy T. Liu (andi611) ]
"""*********************************************************************************************"""


import logging
import os
from .expert import UpstreamExpert as _UpstreamExpert

logger = logging.getLogger(__name__)


def mlm_custom(
    ckpt: str,
    **kwargs,
):
    assert os.path.isdir(ckpt)
    return _UpstreamExpert(ckpt, **kwargs)


def mlm_local(*args, **kwargs):
    return mlm_custom(*args, **kwargs)


def mlm(*args, **kwargs):
    """
    The default model - Base
        refresh (bool): whether to download ckpt/config again if existed
    """
    return mlm_base(*args, **kwargs)


def mlm_base(**kwargs):
    """
    The Base model
        vocab size: 500
        pre-training data: 960 hours
        discrete model: DinoSR
    """
    kwargs["ckpt"] = "/mnt/andy9_liu/model/libri-960-dinosr-mlm-500/"
    return mlm_custom(**kwargs)
