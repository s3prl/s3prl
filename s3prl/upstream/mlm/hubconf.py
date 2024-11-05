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
    return mlm_base_dinosr(*args, **kwargs)

def mlm_base_dinosr(**kwargs):
    """
    The Base model -
        weight_decay of 0.03
        vocab size: 500
        pre-training data: 960 hours
        discrete model: DinoSR
    """
    kwargs["ckpt"] = "/home/ai611/model/libri-960-dinosr-mlm-500/"
    return mlm_custom(**kwargs)

def mlm_base_hubert(**kwargs):
    """
    The Base model -
        weight_decay of 0.03
        vocab size: 500
        pre-training data: 960 hours
        discrete model: HuBERT
    """
    kwargs["ckpt"] = "/home/ai611/model/libri-960-hubert-mlm-500/"
    return mlm_custom(**kwargs)

def mlm_base_hubert_1000(**kwargs):
    """
    The Base model -
        weight_decay of 0.03
        vocab size: 1000
        pre-training data: 960 hours
        discrete model: HuBERT
    """
    kwargs["ckpt"] = "/home/ai611/model/libri-960-hubert-mlm-1000/"
    return mlm_custom(**kwargs)

def mlm_base_hubert_2x(**kwargs):
    """
    The Base model -
        weight_decay of 0.03
        vocab size: 1000
        pre-training data: 960 hours
        discrete model: HuBERT
        max_seq_length: 2x
    """
    kwargs["ckpt"] = "/home/ai611/model/libri-960-hubert-mlm-1000-2x/"
    return mlm_custom(**kwargs)