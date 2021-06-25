# Copyright (c) Facebook, Inc. All Rights Reserved

# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/wav2vec2/hubconf.py ]
#   Synopsis     [ the wav2vec 2.0 torch hubconf ]
#   Author       [ S3PRL / Kushal Lakhotia]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import torch
#-------------#
from utility.download import _urls_to_filepaths
from .expert import UpstreamExpert as _UpstreamExpert


def wav2vec2_local(ckpt, *args, **kwargs):
    """
        The model from local ckpt
            ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, *args, **kwargs)


def wav2vec2_url(ckpt, refresh=False, *args, **kwargs):
    """
        The model from google drive id
            ckpt (str): URL
            refresh (bool): whether to download ckpt/config again if existed
    """
    return wav2vec2_local(_urls_to_filepaths(ckpt, refresh=refresh), *args, **kwargs)


def wav2vec2(refresh=False, *args, **kwargs):
    """
        The default model - Base
            refresh (bool): whether to download ckpt/config again if existed
    """
    return wav2vec2_base_960(refresh=refresh, *args, **kwargs)


def wav2vec2_base_960(refresh=False, *args, **kwargs):
    """
        The Base model
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs['ckpt'] = 'https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt'
    return wav2vec2_url(refresh=refresh, *args, **kwargs)


def wav2vec2_large_960(refresh=False, *args, **kwargs):
    """
        The Large model trained on LibriSpeech 960 hours of data
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs['ckpt'] = 'https://dl.fbaipublicfiles.com/fairseq/wav2vec/libri960_big.pt'
    return wav2vec2_url(refresh=refresh, *args, **kwargs)    


def wav2vec2_large_ll60k(refresh=False, *args, **kwargs):
    """
        The Large model trained on Libri-light 60k hours of data
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs['ckpt'] = 'https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox_new.pt'
    return wav2vec2_url(refresh=refresh, *args, **kwargs)


def wav2vec2_xlsr(refresh=False, *args, **kwargs):
    """
        The wav2vec 2.0 model trained on multilingual presented in https://arxiv.org/abs/2006.13979
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs['ckpt'] = 'https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr_53_56k.pt'
    return wav2vec2_url(refresh=refresh, *args, **kwargs)
